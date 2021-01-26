%% Full Smoluchowski Time Stepping solver for the Fokker-Planck Equation (RK3-CN2)
% Loosely based on Spherefun, this solve uses Double Fourier Sphere (DFS)
% method to tranform the orientational space and time-marched in the DFS 
% space. Time stepping is semi-implicit, with Advection terms in RK3 and 
% Laplacian term in CN2. Implicit matrix inversion is done using the
% Spherefun Helmholtz Solver (only slightly modified to remove
% chebfun-based activiities). 

parpool(2);
%% Setting up
% Parameters
Vc=.05;                       % Swimming Speed (scaled by channel width and Dr) (Pe_s)
Pef=2.5;                      % Flow Peclet Number (Pe_f)
Vsmin=0;                  % Minimum sedimentaion (Vs)
Vsvar=0;                  % Vs_max-Vs_min

diff_const = 1;             % Rotational Diffusion constant
DT=.0;                      % Translational Diffusion constant
beta=2.2;                   % Gyrotactic time scale
AR=1;                      % Aspect Ratio of swimmer (1=spherical) % AR=1.3778790674938353091971374518539773339097820167847;
B=(AR^2-1)/(AR^2+1);        % Bretherton Constant of swimmer (a.k.a. alpha0)

dt = 0.005;                  % Time step
tfinal = 60+dt*2;           % Stopping time
nsteps = ceil(tfinal/dt);   % Number of time steps
m = 12;                     % Spatial discretization - phi (even)
n = 20;                     % Spaptial discretization - theta (even)
Nx_mesh=160;                % Spaptial discretization - x
Nz_mesh=640;                % Spaptial discretization - z

x_width=2;
z_width=8;
epsInit=0.005;

% Run saving settings
saving_rate2=500;
saving_rate3=100;

%Saving to settings struct
settings.beta=beta;
settings.n=n;
settings.m=m;
settings.omg1=0;
settings.omg2=1;
settings.omg3=0;
settings.e12=0;
settings.e22=0;
settings.e23=0;

%% 2D-domain Meshing
dx=x_width/(Nx_mesh);
x=-x_width/2:dx:x_width/2-dx;
dz=z_width/(Nz_mesh);
z=(-z_width/2:dz:z_width/2-dz)';

%Dx
Rdxx=spdiags(ones(Nx_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],Nx_mesh,Nx_mesh);
Rdxx=spdiags(ones(Nx_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-Nx_mesh+3:-1:-Nx_mesh+1 Nx_mesh-1:-1:Nx_mesh-3],Rdxx);
Rdxx=Rdxx/dx;
Rdx=gpuArray(kron(Rdxx,speye(Nz_mesh)));
Rd2xx=spdiags(ones(Nx_mesh,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90],[3:-1:-3],Nx_mesh,Nx_mesh);
Rd2xx=spdiags(ones(Nx_mesh,1)*[1/90 -3/20 3/2 3/2 -3/20 1/90],[-Nx_mesh+3:-1:-Nx_mesh+1 Nx_mesh-1:-1:Nx_mesh-3],Rd2xx);
Rd2xx=Rd2xx/dx/dx;
Rd2x=gpuArray(kron(Rd2xx,speye(Nz_mesh)));

%Dz
Rdzz=spdiags(ones(Nz_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],Nz_mesh,Nz_mesh);
Rdzz=spdiags(ones(Nz_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-Nz_mesh+3:-1:-Nz_mesh+1 Nz_mesh-1:-1:Nz_mesh-3],Rdzz);
Rdzz=Rdzz/dz;
Rdz=gpuArray(kron(speye(Nx_mesh),Rdzz));
Rd2zz=spdiags(ones(Nz_mesh,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90],[3:-1:-3],Nz_mesh,Nz_mesh);
Rd2zz=spdiags(ones(Nz_mesh,1)*[1/90 -3/20 3/2 3/2 -3/20 1/90],[-Nz_mesh+3:-1:-Nz_mesh+1 Nz_mesh-1:-1:Nz_mesh-3],Rd2zz);
Rd2zz=Rd2zz/dz/dz;
Rd2z=gpuArray(kron(speye(Nx_mesh),Rd2zz));

%% Shear Profile
U_profile=zeros(1,Nx_mesh*Nz_mesh);   % W(x)=cos(pi x)+1
W_profile=reshape(ones(size(z))*x,1,[])*Pef;

omg2_profile=reshape(ones(size(z))*-ones(size(x))*Pef/2,1,[]); % -.5*dW(x)/dx=-pi*sin(pi x)/2

e11_profile=zeros(1,Nz_mesh*Nx_mesh);
e13_profile=reshape(ones(size(z))*ones(size(x))*Pef/2,1,[]); % .5*dW(x)/dx=-pi*sin(pi x)/2
e33_profile=zeros(1,Nz_mesh*Nx_mesh);

%% RK3 coeff and constants
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];

% Preparing Constants
K2 = (1/(dt*diff_const));         % Helmholtz frequency for BDF1

%% Initialising Matrices
% Surface Integrals
arr=[-n/2:n/2-1];
fac=2./(1-arr.^2);
if mod(n/2,2)
    fac(1:2:end)=0;
    fac(n/2)=0;
    fac(n/2+2)=0;
else
    fac(2:2:end)=0;
end
Mint=gpuArray(kron(fac,[zeros(1,m/2) 1 zeros(1,m/2-1)]));
MintSq=Mint*Mint';
settings.Mint=Mint;
settings.MintSq=MintSq;
Kp=0.001;
settings.Kp=Kp/settings.MintSq/diff_const/dt;
Kp=settings.Kp;

% Advection
% Madv=adv_mat(settings);
% Mvor=gpuArray(complex(adv_vor_mat(settings)));
% settings.e11=1;settings.e13=0;settings.e33=0;
% Me11=gpuArray(complex(B*adv_strain_mat(settings)));
% settings.e11=0;settings.e13=1;settings.e33=0;
% Me13=gpuArray(complex(B*adv_strain_mat(settings)));
% settings.e11=0;settings.e13=0;settings.e33=1;
% Me33=gpuArray(complex(B*adv_strain_mat(settings)));
% Mgyro=gpuArray(complex(settings.beta*adv_gyro_mat(settings)));
%For strange behaviour in MATLAB ver < R2020
Mvor=gpuArray(sparse(complex(full(adv_vor_mat(settings)))));
settings.e11=1;settings.e13=0;settings.e33=0;
Me11=gpuArray(sparse(complex(full(B*adv_strain_mat(settings)))));
settings.e11=0;settings.e13=1;settings.e33=0;
Me13=gpuArray(sparse(complex(full(B*adv_strain_mat(settings)))));
settings.e11=0;settings.e13=0;settings.e33=1;
Me33=gpuArray(sparse(complex(full(B*adv_strain_mat(settings)))));
Mgyro=gpuArray(sparse(complex(full(settings.beta*adv_gyro_mat(settings)))));

%Laplacian
Mlap=gpuArray(sparse(complex(full(lap_mat(settings)))));
helm=helmholtz_genGPU( n, m);
helm_inv_k1=helmholtz_precalGPU( -K2/alpha(1),helm);
helm_inv_k2=helmholtz_precalGPU( -K2/alpha(2),helm);
helm_inv_k3=helmholtz_precalGPU( -K2/alpha(3),helm);

%p1
Mp1 = gpuArray(complex(kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m))));
%p3
Mp3 = gpuArray(sparse(complex(full(kron(spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n),speye(m)))))); 
%p3p3
Mp3sq = gpuArray(sparse(complex(full(kron(spdiags(ones(n,1)*[.25,.5,.25], [-2 0 2], n, n),speye(m))))));
%p1p3
Mp1p3 = kron(spdiags(.25i*ones(n,1)*[-1,1], [-2 2], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));

%Swimming and sedimentation
MSwim_dz=Vc*Mp3-Vsmin*gpuArray(speye(n*m))-Vsvar*Mp3sq;
MSwim_dx=Vc*Mp1-Vsvar*Mp1p3;

int_const=1.;
settings.int_const=int_const;

%% Initial e-space only run
ucoeff=zeros(n*m,1);ucoeff(m*n/2+m/2+1,:)=1/4/pi;
ucoeff=gpuArray(ucoeff);
Nint_loc=real(Mint*ucoeff*2*pi);

for i = 1:nsteps
    %% RK step 1
    k=1;
    
    adv_coeff=Pef/2*((Me13-Mvor)*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    adv_p_coeff=adv_coeff;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/diff_const/alpha(k)*gamma(k))*adv_p_coeff...
             -(Kp/alpha(k)*(int_const-Nint_loc))*Mint'.*ucoeff;
         
    F=permute(reshape(rhs_coeff,helm.n,helm.m),[2 1]);
    int_constj = pagefun(@mtimes,(-alpha(k)/K2)*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,(-alpha(k)/K2)*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k1*reshape(F,helm.n*helm.m,1);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n),[2 1]),helm.n*helm.m,1);  
    
    Nint_loc=real(Mint*ucoeff*2*pi);
    
    %% RK step 2
    k=2;

    adv_coeff=Pef/2*((Me13-Mvor)*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;

    adv_comb_coeff=adv_coeff;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/diff_const/alpha(k)*gamma(k))*adv_comb_coeff+(1/diff_const/alpha(k)*rho(k))*adv_p_coeff...
            -(Kp/alpha(k)*(int_const-Nint_loc))*Mint'.*ucoeff;
         
    F=permute(reshape(rhs_coeff,helm.n,helm.m),[2 1]);
    int_constj = pagefun(@mtimes,(-alpha(k)/K2)*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,(-alpha(k)/K2)*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k2*reshape(F,helm.n*helm.m,1);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n),[2 1]),helm.n*helm.m,1);     

    Nint_loc=real(Mint*ucoeff*2*pi);
    
    %% RK step 3
    k=3;
    adv_p_coeff=adv_comb_coeff;
    
    adv_coeff=Pef/2*((Me13-Mvor)*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;

    adv_comb_coeff=adv_coeff;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/diff_const/alpha(k)*gamma(k))*adv_comb_coeff+(1/diff_const/alpha(k)*rho(k))*adv_p_coeff...
            -(Kp/alpha(k)*(int_const-Nint_loc))*Mint'.*ucoeff;
         
    F=permute(reshape(rhs_coeff,helm.n,helm.m),[2 1]);
    int_constj = pagefun(@mtimes,(-alpha(k)/K2)*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,(-alpha(k)/K2)*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k3*reshape(F,helm.n*helm.m,1);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n),[2 1]),helm.n*helm.m,1);     

    Nint_loc=real(Mint*ucoeff*2*pi);

end
%% Initial Condition
% oneD= load('smol_pBC_2-2beta_0B_0Vsv_0-25Vc_0DT_2Pef_cospi_cd100_m16_n20_dt0-01_tf100-02GPU.mat','ucoeff');
ucoeff0=gather(ucoeff)*ones(1,Nx_mesh*Nz_mesh);
norm_distrx=exp(-x.^2/epsInit);
norm_distrx=norm_distrx/(sum(norm_distrx)*dx);
norm_distrz=exp(-z.^2/epsInit);
norm_distrz=norm_distrz/(sum(norm_distrz)*dz);

ucoeff0=ucoeff0.*reshape(norm_distrz*norm_distrx,1,[]);

%% Initialise Recorded values
cell_den=NaN(floor(nsteps/saving_rate3),Nz_mesh*Nx_mesh);

%% Time-Stepping (RK3-CN2)
ucoeff=gpuArray(complex(ucoeff0));
% ucoeff_previous=gpuArray(complex(NaN(n*m,Nz_mesh*Nx_mesh,3)));

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dz*dx;

for i = 1:nsteps
    %% RK step 1
    k=1;
    dxu_coeff=ucoeff*Rdx;
    dx2u_coeff=ucoeff*Rd2x;
    dzu_coeff=ucoeff*Rdz;
    dz2u_coeff=ucoeff*Rd2z;
    
    adv_coeff=omg2_profile.*(Mvor*ucoeff)+e11_profile.*(Me11*ucoeff)...
        +e13_profile.*(Me13*ucoeff)+e33_profile.*(Me33*ucoeff)...
        +Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    uadv_coeff=U_profile.*dxu_coeff+W_profile.*dzu_coeff;
    swim_coeff=MSwim_dx*dxu_coeff+MSwim_dz*dzu_coeff;
    DT_coeff=DT*(dx2u_coeff+dz2u_coeff);
    adv_p_coeff=adv_coeff+swim_coeff+uadv_coeff-DT_coeff;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/diff_const/alpha(k)*gamma(k))*adv_p_coeff...
             -(Kp/alpha(k)*(int_const-Nint_loc))*Mint'.*ucoeff;
         
    F=permute(reshape(rhs_coeff,helm.n,helm.m,Nz_mesh*Nx_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,(-alpha(k)/K2)*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,(-alpha(k)/K2)*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k1*reshape(F,helm.n*helm.m,Nz_mesh*Nx_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,Nz_mesh*Nx_mesh),[2 1 3]),helm.n*helm.m,Nz_mesh*Nx_mesh);  
    
    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dz*dx;
    
    %% RK step 2
    k=2;
    dxu_coeff=ucoeff*Rdx;
    dx2u_coeff=ucoeff*Rd2x;
    dzu_coeff=ucoeff*Rdz;
    dz2u_coeff=ucoeff*Rd2z;
    
    adv_coeff=omg2_profile.*(Mvor*ucoeff)+e11_profile.*(Me11*ucoeff)...
        +e13_profile.*(Me13*ucoeff)+e33_profile.*(Me33*ucoeff)...
        +Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    uadv_coeff=U_profile.*dxu_coeff+W_profile.*dzu_coeff;
    swim_coeff=MSwim_dx*dxu_coeff+MSwim_dz*dzu_coeff;
    DT_coeff=DT*(dx2u_coeff+dz2u_coeff);
    adv_comb_coeff=adv_coeff+swim_coeff+uadv_coeff-DT_coeff;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/diff_const/alpha(k)*gamma(k))*adv_comb_coeff+(1/diff_const/alpha(k)*rho(k))*adv_p_coeff...
            -(Kp/alpha(k)*(int_const-Nint_loc))*Mint'.*ucoeff;
         
    F=permute(reshape(rhs_coeff,helm.n,helm.m,Nz_mesh*Nx_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,(-alpha(k)/K2)*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,(-alpha(k)/K2)*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k2*reshape(F,helm.n*helm.m,Nz_mesh*Nx_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,Nz_mesh*Nx_mesh),[2 1 3]),helm.n*helm.m,Nz_mesh*Nx_mesh);     

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dz*dx;
    
    %% RK step 3
    k=3;
    dxu_coeff=ucoeff*Rdx;
    dx2u_coeff=ucoeff*Rd2x;
    dzu_coeff=ucoeff*Rdz;
    dz2u_coeff=ucoeff*Rd2z;
    adv_p_coeff=adv_comb_coeff;
    
    adv_coeff=omg2_profile.*(Mvor*ucoeff)+e11_profile.*(Me11*ucoeff)...
        +e13_profile.*(Me13*ucoeff)+e33_profile.*(Me33*ucoeff)...
        +Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    uadv_coeff=U_profile.*dxu_coeff+W_profile.*dzu_coeff;
    swim_coeff=MSwim_dx*dxu_coeff+MSwim_dz*dzu_coeff;
    DT_coeff=DT*(dx2u_coeff+dz2u_coeff);
    adv_comb_coeff=adv_coeff+swim_coeff+uadv_coeff-DT_coeff;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/diff_const/alpha(k)*gamma(k))*adv_comb_coeff+(1/diff_const/alpha(k)*rho(k))*adv_p_coeff...
            -(Kp/alpha(k)*(int_const-Nint_loc))*Mint'.*ucoeff;
         
    F=permute(reshape(rhs_coeff,helm.n,helm.m,Nz_mesh*Nx_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,(-alpha(k)/K2)*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,(-alpha(k)/K2)*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k3*reshape(F,helm.n*helm.m,Nz_mesh*Nx_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,Nz_mesh*Nx_mesh),[2 1 3]),helm.n*helm.m,Nz_mesh*Nx_mesh);     

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dz*dx;
    
    %% Saving for Post-Processing
    %  Plot/Save the solution every saving_rate
    if ( mod(i, saving_rate2) == 0 )
        ufull_save=gather(ucoeff);
        t=i*dt;
        save(['smol_pBC_2D_t' num2str(t) '.mat'],'t','ufull_save');
    end
    
    %% On-the-go-Post-Processing
     if ( mod(i, saving_rate3) == 0 )
        cellden_temp=real(Mint*ucoeff*2*pi);
%         f=gather(ucoeff./cellden_temp);
        cell_den(i/saving_rate3,:)=gather(cellden_temp);
%         PS_feval(i/saving_rate3)=parfeval(@PS_transformed2D,24,f,CPUPS);
    end 
%     if ( mod(i, saving_rate3) == (saving_rate3-2) )
%         ucoeff_previous(:,:,1)=gather(ucoeff);
%     end 
%     if ( mod(i, saving_rate3) == (saving_rate3-1) )
%         ucoeff_previous(:,:,2)=gather(ucoeff);
%     end 
%     if ( mod(i, saving_rate3) == 1 ) && i~=1
%         ucoeff_previous(:,:,3)=gather(ucoeff);
%     end 
%     if ( mod(i, saving_rate3) == 2 ) && i~=2
%         unsteady_RHS=((-gather(ucoeff)./(real(Mint*ucoeff*2*pi))...
%             + ucoeff_previous(:,:,1)./(real(Mint*ucoeff_previous(:,:,1)*2*pi)))/12 ...
%             +(ucoeff_previous(:,:,3)./(real(Mint*ucoeff_previous(:,:,3)*2*pi))...
%             -ucoeff_previous(:,:,2)./(real(Mint*ucoeff_previous(:,:,2)*2*pi)))*(2/3))/dt;
%         
%         temp=sum(bsxfun(@times,Linv,reshape([unsteady_RHS;zeros(1,Nz_mesh*Nx_mesh,'gpuArray')],1,n*m+1,Nz_mesh*Nx_mesh)),2);
%         f_unsteady=reshape(temp(1:n*m,1,:),n*m,Nz_mesh*Nx_mesh);
%         
%         Vux((i-2)/saving_rate3,:)=gather(Mint*Mp1*(reshape(f_unsteady,n*m,Nz_mesh*Nx_mesh))*(2*pi));
%         Vuz((i-2)/saving_rate3,:)=gather(Mint*Mp3*(reshape(f_unsteady,n*m,Nz_mesh*Nx_mesh))*(2*pi));
% 
%     end 
disp([num2str(i) '/' num2str(nsteps)]);
end

%% Surface Integral Conservation check
% t1=dt*saving_rate1:dt*saving_rate1:tfinal;
t2=dt*saving_rate2:dt*saving_rate2:tfinal;
t3=dt*saving_rate3:dt*saving_rate3:tfinal;

Nint=sum(cell_den,2)*dz*dx;

%% Saving Data
omg2_profile=gather(omg2_profile);
e11_profile=gather(e11_profile);
e13_profile=gather(e13_profile);
e33_profile=gather(e33_profile);
U_profile=gather(U_profile);
W_profile=gather(W_profile);
Kp=gather(Kp);
% ucoeff=gather(ucoeff);
ex_file_name=['smol_pBC_2D_' num2str(epsInit) 'epsInit_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsmin) 'Vsm_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_Nx' num2str(Nx_mesh) 'Nz' num2str(Nz_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];
ex_file_name=replace(ex_file_name,'.','-');

save([ex_file_name 'GPU.mat'],...
    'n','m','Nx_mesh','Nz_mesh','nsteps','Vc','Pef','beta','diff_const','DT','B','Vsmin','Vsvar',...
    'omg2_profile','e11_profile','e13_profile','e33_profile','U_profile','W_profile',...
    'dt','tfinal','settings','Kp','x','dx','z','dz','x_width','z_width','epsInit',...
    'saving_rate2','saving_rate3',...
    't2','t3','Nint','cell_den',...
    'ucoeff','ucoeff0',...
    '-v7.3');

exit
