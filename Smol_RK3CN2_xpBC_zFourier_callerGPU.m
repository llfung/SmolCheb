%% Full Smoluchowski Time Stepping solver for the Fokker-Planck Equation (RK3-CN2)
% Loosely based on Spherefun, this solve uses Double Fourier Sphere (DFS)
% method to tranform the orientational space and time-marched in the DFS 
% space. Time stepping is semi-implicit, with Advection terms in RK3 and 
% Laplacian term in CN2. Implicit matrix inversion is done using the
% Spherefun Helmholtz Solver (only slightly modified to remove
% chebfun-based activiities). 
gpuDevice(1);
% parpool(2);
%% Setting up
% Parameters
Vc=.025;                       % Swimming Speed (scaled by channel width and Dr) (Pe_s)
Pef=.5;                      % Flow Peclet Number (Pe_f)
Vsmin=0;                  % Minimum sedimentaion (Vs)
Vsvar=0;                  % Vs_max-Vs_min

diff_const = 1;             % Rotational Diffusion constant
DT=.0;                      % Translational Diffusion constant
beta=2.2;                   % Gyrotactic time scale
% AR=1;                      % Aspect Ratio of swimmer (1=spherical) % AR=1.3778790674938353091971374518539773339097820167847;
% B=(AR^2-1)/(AR^2+1);        % Bretherton Constant of swimmer (a.k.a. alpha0)
B=0;

dt = 0.005;                  % Time step
tfinal = 40;%+dt*2;           % Stopping time
nsteps = ceil(tfinal/dt);   % Number of time steps
m = 8;                     % Spatial discretization - phi (even)
n = 16;                     % Spaptial discretization - theta (even)
Nx_mesh=128;                 % Spaptial discretization - x
Nz_mesh=256;                 % Spectral discretization - z

x_width=2;
z_width=8;
epsInit=0.005;
alphaK=2*pi/z_width;



omg=[0,-1,0];               % Vorticity direction (1,2,3) 

% Run saving settings
saving_rate1=100000000;
saving_rate2=400;
saving_rate3=40;

% x_sav_location=[1 11 21 33 24 3 42 45 48];
x_sav_location=[1 11 26 31 51];

%Saving to settings struct
settings.beta=beta;
settings.n=n;
settings.m=m;
settings.omg1=omg(1);
settings.omg2=omg(2);
settings.omg3=omg(3);
settings.e11=0;
settings.e12=0;
settings.e13=1;
settings.e22=0;
settings.e23=0;
settings.e33=0;

%% 2D-domain Meshing
dx=x_width/(Nx_mesh);
x=-x_width/2:dx:x_width/2-dx;
dz=z_width/(Nz_mesh);
z=(-z_width/2:dz:z_width/2-dz);

%% Shear Profile
W_profile=gpuArray(repmat(x*Pef,1,Nz_mesh));   % W(x)=-cos(pi x)-1
U_profile=gpuArray(zeros(1,Nx_mesh*Nz_mesh));
S_profile= ones(size(x))*Pef/2; % .5*dW(x)/dx= pi*sin(pi x)/2
% S_profile(1)=0;

% S_profile=x*Pef; % W(x)=-(1-x^2)
% S_profile=Pef/2*ones(size(x)); % W(x)=x

S_profile=gpuArray(repmat(S_profile,1,Nz_mesh));


%% RK3 coeff and constants
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];

% Preparing Constants
K2 = (1/(dt*diff_const));         % Helmholtz frequency for BDF1

%Dx
Rdx=spdiags(ones(Nx_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],Nx_mesh,Nx_mesh);
Rdx=spdiags(ones(Nx_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-Nx_mesh+3:-1:-Nx_mesh+1 Nx_mesh-1:-1:Nx_mesh-3],Rdx);
Rdx=gpuArray(kron(speye(Nz_mesh),Rdx/dx));
Rd2x=spdiags(ones(Nx_mesh,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90],[3:-1:-3],Nx_mesh,Nx_mesh);
Rd2x=spdiags(ones(Nx_mesh,1)*[1/90 -3/20 3/2 3/2 -3/20 1/90],[-Nx_mesh+3:-1:-Nx_mesh+1 Nx_mesh-1:-1:Nx_mesh-3],Rd2x);
Rd2x=gpuArray(kron(speye(Nz_mesh),Rd2x/dx/dx));

%Dz
rowDz=gpuArray(kron(ifftshift((-(Nz_mesh/2):(Nz_mesh/2)-1)*1i*alphaK),ones(1,Nx_mesh)));
rowDzz=gpuArray(kron(ifftshift((-(Nz_mesh/2):(Nz_mesh/2)-1).^2*alphaK^2*(-1)),ones(1,Nx_mesh)));
Nint_row=gpuArray([ones(1,Nx_mesh)*dx*z_width zeros(1,(Nz_mesh-1)*Nx_mesh)]');

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
% Mgyro=gpuArray(complex(settings.beta*adv_gyro_mat(settings)));
%For strange behaviour in MATLAB ver < R2020
Mvor=gpuArray(sparse(complex(full(adv_vor_mat(settings)+B*adv_strain_mat(settings)))));
Mgyro=gpuArray(sparse(complex(full(settings.beta*adv_gyro_mat(settings)))));

%Laplacian
Mlap=gpuArray(sparse(complex(full(lap_mat(settings)))));
helm=helmholtz_genGPU( n, m);
helm_inv_k1=helmholtz_precalGPU( -K2/alpha(1),helm);
helm_inv_k2=helmholtz_precalGPU( -K2/alpha(2),helm);
helm_inv_k3=helmholtz_precalGPU( -K2/alpha(3),helm);

helm_CPUPS=helmholtz_gen( n, m);
helm_CPUPS.helm_inv_k1=helmholtz_precal( -K2/alpha(1),helm_CPUPS);
helm_CPUPS.helm_inv_k2=helmholtz_precal( -K2/alpha(2),helm_CPUPS);
helm_CPUPS.helm_inv_k3=helmholtz_precal( -K2/alpha(3),helm_CPUPS);
helm_CPUPS.dt=dt;

%p1
Mp1 = gpuArray(complex(kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m))));
%p3
Mp3 = gpuArray(sparse(complex(full(kron(spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n),speye(m))))));
%p3p3
Mp3sq = gpuArray(sparse(complex(full(kron(spdiags(ones(n,1)*[.25,.5,.25], [-2 0 2], n, n),speye(m))))));
%p1p3
Mp1p3 = gpuArray(sparse(complex(full(kron(spdiags(.25i*ones(n,1)*[-1,1], [-2 2], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m))))));

%Swimming and sedimentation
MSwim_dx=Vc*Mp1-Vsvar*Mp1p3;
MSwim_dz=Vc*Mp3-Vsmin*gpuArray(speye(n*m))-Vsvar*Mp3sq;

int_const=1.;
settings.int_const=int_const;

%% Initial e-space only run
ucoeff=zeros(n*m,1);ucoeff(m*n/2+m/2+1,:)=1/4/pi;
ucoeff=gpuArray(ucoeff);
Nint_loc=real(Mint*ucoeff*2*pi);

for i = 1:nsteps
    %% RK step 1
    k=1;

    adv_coeff=Pef/2*(Mvor*ucoeff)+Mgyro*ucoeff;
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

    adv_coeff=Pef/2*(Mvor*ucoeff)+Mgyro*ucoeff;
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
    
    adv_coeff=Pef/2*(Mvor*ucoeff)+Mgyro*ucoeff;
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
norm_distrx=exp(-x.^2/epsInit);
norm_distrx=norm_distrx/(sum(norm_distrx)*dx);
norm_distrz=exp(-z.^2/epsInit);
norm_distrz=norm_distrz/(sum(norm_distrz)*dz);
norm_distrz_T=fft(norm_distrz)/Nz_mesh;

ucoeff0=gather(ucoeff)*reshape(transpose(norm_distrx)*norm_distrz_T,1,[]);

%% Initialise Recorded values
cell_den=NaN(floor(nsteps/saving_rate3),Nx_mesh*Nz_mesh);

ufull_save=NaN(n*m,Nx_mesh*Nz_mesh,floor(nsteps/saving_rate2));
fdt_full_save=NaN(n*m,Nx_mesh*Nz_mesh,floor(nsteps/saving_rate2));
fndt_full_save=NaN(n*m,Nx_mesh*Nz_mesh,floor(nsteps/saving_rate2));

u_xloc_save=NaN(n*m,floor(nsteps/saving_rate1),length(x_sav_location));


%% Time-Stepping (RK3-CN2)
ucoeff=gpuArray(complex(ucoeff0));
adv_p_coeff   =gpuArray(complex(zeros(n*m,Nx_mesh*Nz_mesh)));
adv_comb_coeff=gpuArray(complex(zeros(n*m,Nx_mesh*Nz_mesh)));
ucoeff_previous=gpuArray(complex(NaN(n*m,Nx_mesh*Nz_mesh,3)));
% ucoeff_previous2=gpuArray(complex(NaN(n*m,N_mesh,3)));

    cell_den_loc=(Mint*ucoeff*2*pi);
    Nint_loc=cell_den_loc*Nint_row;

for i = 1:nsteps
    %% RK step 1
    k=1;
    dxu_coeff=ucoeff*Rdx;
    dzu_coeff=ucoeff.*rowDz;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    uadv_coeff=U_profile.*dxu_coeff+W_profile.*dzu_coeff;
    swim_coeff=MSwim_dx*dxu_coeff+MSwim_dz*dzu_coeff;
    DT_coeff=DT*(ucoeff*Rd2x+ucoeff.*rowDzz);
    adv_p_coeff=adv_coeff+swim_coeff+uadv_coeff-DT_coeff;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/diff_const/alpha(k)*gamma(k))*adv_p_coeff...
             -(Kp/alpha(k)*(int_const-Nint_loc))*Mint'.*ucoeff;
         
    F=permute(reshape(rhs_coeff,helm.n,helm.m,Nx_mesh*Nz_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,(-alpha(k)/K2)*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,(-alpha(k)/K2)*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k3*reshape(F,helm.n*helm.m,Nx_mesh*Nz_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,Nx_mesh*Nz_mesh),[2 1 3]),helm.n*helm.m,Nx_mesh*Nz_mesh);
    
    cell_den_loc=(Mint*ucoeff*2*pi);
    Nint_loc=cell_den_loc*Nint_row;
    
    %% RK step 2
    k=2;
    dxu_coeff=ucoeff*Rdx;
    dzu_coeff=ucoeff.*rowDz;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    uadv_coeff=U_profile.*dxu_coeff+W_profile.*dzu_coeff;
    swim_coeff=MSwim_dx*dxu_coeff+MSwim_dz*dzu_coeff;
    DT_coeff=DT*(ucoeff*Rd2x+ucoeff.*rowDzz);
    adv_comb_coeff=adv_coeff+swim_coeff+uadv_coeff-DT_coeff;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/diff_const/alpha(k)*gamma(k))*adv_comb_coeff+(1/diff_const/alpha(k)*rho(k))*adv_p_coeff...
            -(Kp/alpha(k)*(int_const-Nint_loc))*Mint'.*ucoeff;
         
    F=permute(reshape(rhs_coeff,helm.n,helm.m,Nx_mesh*Nz_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,(-alpha(k)/K2)*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,(-alpha(k)/K2)*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k3*reshape(F,helm.n*helm.m,Nx_mesh*Nz_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,Nx_mesh*Nz_mesh),[2 1 3]),helm.n*helm.m,Nx_mesh*Nz_mesh);
    
    cell_den_loc=(Mint*ucoeff*2*pi);
    Nint_loc=cell_den_loc*Nint_row;
    
    %% RK step 3
    k=3;
    adv_p_coeff=adv_comb_coeff;
    dxu_coeff=ucoeff*Rdx;
    dzu_coeff=ucoeff.*rowDz;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    uadv_coeff=U_profile.*dxu_coeff+W_profile.*dzu_coeff;
    swim_coeff=MSwim_dx*dxu_coeff+MSwim_dz*dzu_coeff;
    DT_coeff=DT*(ucoeff*Rd2x+ucoeff.*rowDzz);
    adv_comb_coeff=adv_coeff+swim_coeff+uadv_coeff-DT_coeff;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/diff_const/alpha(k)*gamma(k))*adv_comb_coeff+(1/diff_const/alpha(k)*rho(k))*adv_p_coeff...
            -(Kp/alpha(k)*(int_const-Nint_loc))*Mint'.*ucoeff;
         
    F=permute(reshape(rhs_coeff,helm.n,helm.m,Nx_mesh*Nz_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,(-alpha(k)/K2)*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,(-alpha(k)/K2)*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k3*reshape(F,helm.n*helm.m,Nx_mesh*Nz_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,Nx_mesh*Nz_mesh),[2 1 3]),helm.n*helm.m,Nx_mesh*Nz_mesh);
    
    cell_den_loc=(Mint*ucoeff*2*pi);
    Nint_loc=cell_den_loc*Nint_row;
    
    %% Saving for Post-Processing
%     if ( mod(i, saving_rate1) == 0 )
%         for j=1:length(x_sav_location)
%             u_xloc_save(:,i/saving_rate1,j)=gather(ucoeff(:,x_sav_location(j)));
%         end
%     end
%     
    % Plot/Save the solution every saving_rate
    if ( mod(i, saving_rate2) == 0 )
        ufull_save=gather(ucoeff);
        t=i*dt;
        save(['smol_pBC_2D_t' num2str(t) '.mat'],'t','ufull_save');
    end
%     if ( mod(i, saving_rate2) == 2 )&& i~=2 
%         fdt_full_save(:,:,(i-2)/saving_rate2)=gather((-ucoeff./(real(Mint*ucoeff*2*pi))...
%             + ucoeff_previous2(:,:,1)./(real(Mint*ucoeff_previous2(:,:,1)*2*pi)))/12 ...
%             +(ucoeff_previous2(:,:,3)./(real(Mint*ucoeff_previous2(:,:,3)*2*pi))...
%             -ucoeff_previous2(:,:,2)./(real(Mint*ucoeff_previous2(:,:,2)*2*pi)))*(2/3))/dt;
%         fndt_full_save(:,:,(i-2)/saving_rate2)=gather((-ucoeff...
%             + ucoeff_previous2(:,:,1))/12 ...
%             +(ucoeff_previous2(:,:,3)...
%             -ucoeff_previous2(:,:,2))*(2/3))/dt;
%     end
%     if ( mod(i, saving_rate2) == 1 )&& i~=1 
%         ucoeff_previous2(:,:,3)=ucoeff;
%     end
%     if ( mod(i, saving_rate2) == saving_rate2-1 )
%         ucoeff_previous2(:,:,2)=ucoeff;
%     end
%     if ( mod(i, saving_rate2) == saving_rate2-2 )
%         ucoeff_previous2(:,:,1)=ucoeff;
%     end
    
    %% On-the-go-Post-Processing
     if ( mod(i, saving_rate3) == 0 )
        cell_den(i/saving_rate3,:)=gather(Mint*ucoeff*2*pi);
        disp([num2str(i) '/' num2str(nsteps)]);
    end 
%     if ( mod(i, saving_rate3) == (saving_rate3-2) )
%         ucoeff_previous(:,:,1)=ucoeff;
%     end 
%     if ( mod(i, saving_rate3) == (saving_rate3-1) )
%         ucoeff_previous(:,:,2)=ucoeff;
%     end 
%     if ( mod(i, saving_rate3) == 1 ) && i~=1
%         ucoeff_previous(:,:,3)=ucoeff;
%     end 
%     if ( mod(i, saving_rate3) == 2 ) && i~=2
%         unsteady_RHS=((-ucoeff./(real(Mint*ucoeff*2*pi))...
%             + ucoeff_previous(:,:,1)./(real(Mint*ucoeff_previous(:,:,1)*2*pi)))/12 ...
%             +(ucoeff_previous(:,:,3)./(real(Mint*ucoeff_previous(:,:,3)*2*pi))...
%             -ucoeff_previous(:,:,2)./(real(Mint*ucoeff_previous(:,:,2)*2*pi)))*(2/3))/dt;
%         
%         temp=sum(bsxfun(@times,Linv,reshape([unsteady_RHS;zeros(1,N_mesh,'gpuArray')],1,n*m+1,N_mesh)),2);
%         f_unsteady=reshape(temp(1:n*m,1,:),n*m,N_mesh);
%         
%         Vux((i-2)/saving_rate3,:)=gather(Mint*Mp1*(reshape(f_unsteady,n*m,N_mesh))*(2*pi));
%         Vuz((i-2)/saving_rate3,:)=gather(Mint*Mp3*(reshape(f_unsteady,n*m,N_mesh))*(2*pi));
% 
%     end 
end

%% Surface Integral Conservation check
t1=dt*saving_rate1:dt*saving_rate1:tfinal;
t2=dt*saving_rate2:dt*saving_rate2:tfinal;
t3=dt*saving_rate3:dt*saving_rate3:tfinal;

Nint=Nint_loc;
% Nint=NaN(size(t3));
% for i=1:length(t3)
%     Nint(i)=cheb.cheb_int(cell_den(i,:)');
% end

S_profile=gather(S_profile);
Kp=gather(Kp);
ucoeff=gather(ucoeff);
ex_file_name=['smol_pBC_2D' num2str(epsInit) 'epsInit_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_homo_dx_' num2str(Nx_mesh) 'dz_' num2str(Nz_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];
% ex_file_name=['smol_pBC_2D_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_homo_dx_' num2str(Nx_mesh) 'dz_' num2str(Nz_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];

ex_file_name=replace(ex_file_name,'.','-');

save([ex_file_name 'GPU.mat'],...
    'n','m','Nx_mesh','Nz_mesh','nsteps','S_profile','Vc','Pef','omg','beta','diff_const','DT','B','Vsvar','Vsmin',...
    'dt','tfinal','settings','Kp','x','dx','z','dz','x_width','z_width','epsInit',...
    'saving_rate1','saving_rate2','saving_rate3',...
    't1','t2','t3','Nint','cell_den',...
    'ucoeff','ucoeff0',...
    '-v7.3');

% exit
