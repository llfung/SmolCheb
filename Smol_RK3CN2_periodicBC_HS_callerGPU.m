%% Full Smoluchowski Time Stepping solver for the Fokker-Planck Equation (RK3-CN2)
% Loosely based on Spherefun, this solve uses Double Fourier Sphere (DFS)
% method to tranform the orientational space and time-marched in the DFS 
% space. Time stepping is semi-implicit, with Advection terms in RK3 and 
% Laplacian term in CN2. Implicit matrix inversion is done using the
% Spherefun Helmholtz Solver (only slightly modified to remove
% chebfun-based activiities). 

%% Setting up
% Parameters
Vc=.25;                       % Swimming Speed (scaled by channel width and Dr) (Pe_s)
Pef=1;                      % Flow Peclet Number (Pe_f)
Vsmin=0.;                  % Minimum sedimentaion (Vs)
Vsvar=0.;                  % Vs_max-Vs_min

diff_const = 1;             % Rotational Diffusion constant
DT=.0;                      % Translational Diffusion constant
beta=0.21;                   % Gyrotactic time scale
% AR=15;                      % Aspect Ratio of swimmer (1=spherical) % AR=1.3778790674938353091971374518539773339097820167847;
% B=(AR^2-1)/(AR^2+1);        % Bretherton Constant of swimmer (a.k.a. alpha0)
B=0.31;

dt = 0.001;                  % Time step
tfinal = 100+dt*2;           % Stopping time
nsteps = ceil(tfinal/dt);   % Number of time steps
m = 12;                     % Spatial discretization - phi (even)
n = 24;                     % Spaptial discretization - theta (even)
N_mesh=200;                 % Spaptial discretization - z

omg=[0,1,0];                % Vorticity direction (1,2,3) 

% Run saving settings
saving_rate1=100000000;
saving_rate2=100000000;
saving_rate3=100;

z_sav_location=[1 11 21 33 24 42 45 48 51];
% z_sav_location=[1 21 41 61 81 101 121 129];

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

%% x-domain Meshing
dz=2/(N_mesh);
z=-1:dz:1-dz;
% cheb=chebyshev(N_mesh,2,bc_type.none,tran_type.none);
% z=cheb.col_pt;
% D1=(cheb.D(1))';D2=(cheb.D(2))';

%% Shear Profile
% U_profile=(cos(pi*z)+1)*Pef;   % W(x)=cos(pi x)+1
S_profile=-pi*sin(pi*z)*Pef/2; % .5*dW(x)/dx=-pi*sin(pi x)/2
S_profile(1)=0;

% S_profile=x*Pef; % W(x)=-(1-x^2)
% S_profile=Pef/2*ones(size(x)); % W(x)=x

S_profile=gpuArray(S_profile);

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

%Dx
Rdz=spdiags(ones(N_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],N_mesh,N_mesh);
Rdz=spdiags(ones(N_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-N_mesh+3:-1:-N_mesh+1 N_mesh-1:-1:N_mesh-3],Rdz);
Rdz=gpuArray(Rdz/dz);
Rd2z=spdiags(ones(N_mesh,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90],[3:-1:-3],N_mesh,N_mesh);
Rd2z=spdiags(ones(N_mesh,1)*[1/90 -3/20 3/2 3/2 -3/20 1/90],[-N_mesh+3:-1:-N_mesh+1 N_mesh-1:-1:N_mesh-3],Rd2z);
Rd2z=gpuArray(Rd2z/dz/dz);

%p1
Mp1 = gpuArray(complex(kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m))));
%p3
Mp3 = gpuArray(sparse(complex(full(kron(spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n),speye(m)))))); %e3
%p3p3
Mp3sq = gpuArray(sparse(complex(full(kron(spdiags(ones(n,1)*[.25,.5,.25], [-2 0 2], n, n),speye(m)))))); %e3

%Swimming and sedimentation
MSwim=Vc*Mp3-Vsmin*gpuArray(speye(n*m))-Vsvar*Mp3sq;

%% Initial Condition
int_const=1.;
settings.int_const=int_const;

ucoeff0=zeros(n*m,N_mesh);
ucoeff0(m*n/2+m/2+1,:)=1/8/pi;

%% Initialise Recorded values
cell_den=NaN(floor(nsteps/saving_rate3),N_mesh);

ufull_save=NaN(n*m,N_mesh,floor(nsteps/saving_rate2));
fdt_full_save=NaN(n*m,N_mesh,floor(nsteps/saving_rate2));
fndt_full_save=NaN(n*m,N_mesh,floor(nsteps/saving_rate2));

u_zloc_save=NaN(n*m,floor(nsteps/saving_rate1),length(z_sav_location));

Dxx=NaN(floor(nsteps/saving_rate3),N_mesh);
Dzx=NaN(floor(nsteps/saving_rate3),N_mesh);
Dxz=NaN(floor(nsteps/saving_rate3),N_mesh);
Dzz=NaN(floor(nsteps/saving_rate3),N_mesh);
Vix=NaN(floor(nsteps/saving_rate3),N_mesh);
Viz=NaN(floor(nsteps/saving_rate3),N_mesh);
Vux=NaN(floor(nsteps/saving_rate3),N_mesh);
Vuz=NaN(floor(nsteps/saving_rate3),N_mesh);
ex=NaN(floor(nsteps/saving_rate3),N_mesh);
ez=NaN(floor(nsteps/saving_rate3),N_mesh);
VDTx=NaN(floor(nsteps/saving_rate3),N_mesh);
VDTz=NaN(floor(nsteps/saving_rate3),N_mesh);
DDTxz=NaN(floor(nsteps/saving_rate3),N_mesh);
DDTzz=NaN(floor(nsteps/saving_rate3),N_mesh);

%% Post-Processing GPU Array
Linv=NaN(n*m,n*m+1,N_mesh,'gpuArray');
for j=1:N_mesh
    Le=S_profile(j)*Mvor+Mgyro-Mlap;
    Linv(:,:,j)=pinv([full(Le);full(Mint)]);
end

%% Time-Stepping (RK3-CN2)
ucoeff=gpuArray(complex(ucoeff0));
adv_p_coeff   =gpuArray(complex(zeros(n*m,N_mesh)));
adv_comb_coeff=gpuArray(complex(zeros(n*m,N_mesh)));
ucoeff_previous=gpuArray(complex(NaN(n*m,N_mesh,3)));
% ucoeff_previous2=gpuArray(complex(NaN(n*m,N_mesh,3)));

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dz;

for i = 1:nsteps
    %% RK step 1
    k=1;
    % Par-For Version
    dzu_coeff=ucoeff*Rdz;
    dz2u_coeff=ucoeff*Rd2z;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    swim_coeff=MSwim*dzu_coeff;
    DT_coeff=DT*dz2u_coeff;
    adv_p_coeff=adv_coeff+swim_coeff-DT_coeff;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/diff_const/alpha(k)*gamma(k))*adv_p_coeff...
             -(Kp/alpha(k)*(int_const-Nint_loc))*Mint'.*ucoeff;
         
    F=permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,(-alpha(k)/K2)*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,(-alpha(k)/K2)*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k1*reshape(F,helm.n*helm.m,N_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);  
    
    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dz;
    
    %% RK step 2
    k=2;
    dzu_coeff=ucoeff*Rdz;
    dz2u_coeff=ucoeff*Rd2z;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    swim_coeff=MSwim*dzu_coeff;
    DT_coeff=DT*dz2u_coeff;
    adv_comb_coeff=adv_coeff+swim_coeff-DT_coeff;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/diff_const/alpha(k)*gamma(k))*adv_comb_coeff+(1/diff_const/alpha(k)*rho(k))*adv_p_coeff...
            -(Kp/alpha(k)*(int_const-Nint_loc))*Mint'.*ucoeff;
         
    F=permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,(-alpha(k)/K2)*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,(-alpha(k)/K2)*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k2*reshape(F,helm.n*helm.m,N_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);     

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dz;
    
    %% RK step 3
    k=3;
    dzu_coeff=ucoeff*Rdz;
    dz2u_coeff=ucoeff*Rd2z;
    adv_p_coeff=adv_comb_coeff;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    swim_coeff=MSwim*dzu_coeff;
    DT_coeff=DT*dz2u_coeff;
    adv_comb_coeff=adv_coeff+swim_coeff-DT_coeff;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/diff_const/alpha(k)*gamma(k))*adv_comb_coeff+(1/diff_const/alpha(k)*rho(k))*adv_p_coeff...
            -(Kp/alpha(k)*(int_const-Nint_loc))*Mint'.*ucoeff;
         
    F=permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,(-alpha(k)/K2)*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,(-alpha(k)/K2)*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k3*reshape(F,helm.n*helm.m,N_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);     

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dz;
    
    %% Saving for Post-Processing
%     if ( mod(i, saving_rate1) == 0 )
%         for j=1:length(z_sav_location)
%             u_zloc_save(:,i/saving_rate1,j)=gather(ucoeff(:,z_sav_location(j)));
%         end
%     end
%     
%     %    Plot/Save the solution every saving_rate
%     if ( mod(i, saving_rate2) == 0 )
%         ufull_save(:,:,i/saving_rate2)=gather(ucoeff);
%     end
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
         
        f=ucoeff./cell_den_loc;
%         d2xf=f*Rd2x;
%         dxf=f*Rdx;
        d2zf=f*Rd2z;
        dzf=f*Rdz;
        ex_avg=real(Mint*Mp1*f*(2*pi));
        ez_avg=real(Mint*Mp3*f*(2*pi));
        
        bx_RHS=Mp1*f-ex_avg.*f;
        bz_RHS=Mp3*f-ez_avg.*f;
%         inhomo_p1_RHS=Mp1*(f*Rdx)-(ex_avg*Rdx).*f;
        inhomo_p3_RHS=Mp3*(f*Rdz)-(ez_avg*Rdz).*f;
        
        temp=sum(bsxfun(@times,Linv,reshape([bx_RHS;zeros(1,N_mesh,'gpuArray')],1,n*m+1,N_mesh)),2);
        bx=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        temp=sum(bsxfun(@times,Linv,reshape([bz_RHS;zeros(1,N_mesh,'gpuArray')],1,n*m+1,N_mesh)),2);
        bz=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        temp=sum(bsxfun(@times,Linv,reshape([dzf;zeros(1,N_mesh,'gpuArray')],1,n*m+1,N_mesh)),2);
        b_DT_p3=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        temp=sum(bsxfun(@times,Linv,reshape([inhomo_p3_RHS;zeros(1,N_mesh,'gpuArray')],1,n*m+1,N_mesh)),2);
        f_inhomo_p3=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        temp=sum(bsxfun(@times,Linv,reshape([d2zf;zeros(1,N_mesh,'gpuArray')],1,n*m+1,N_mesh)),2);
        f_a=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        
        Dxx(i/saving_rate3,:)=gather(Mint*(Mp1*reshape(bx,n*m,N_mesh))*(2*pi));
        Dxz(i/saving_rate3,:)=gather(Mint*(Mp1*reshape(bz,n*m,N_mesh))*(2*pi));
        Dzx(i/saving_rate3,:)=gather(Mint*(Mp3*reshape(bx,n*m,N_mesh))*(2*pi));
        Dzz(i/saving_rate3,:)=gather(Mint*(Mp3*reshape(bz,n*m,N_mesh))*(2*pi));
        Vix(i/saving_rate3,:)=gather(Mint*(Mp1*reshape(f_inhomo_p3,n*m,N_mesh))*(2*pi));
        Viz(i/saving_rate3,:)=gather(Mint*(Mp3*reshape(f_inhomo_p3,n*m,N_mesh))*(2*pi));
        ex(i/saving_rate3,:)=gather(ex_avg);
        ez(i/saving_rate3,:)=gather(ez_avg);
                        
        VDTx(i/saving_rate3,:)=gather(Mint*(Mp1*reshape(f_a,n*m,N_mesh))*(2*pi));
        VDTz(i/saving_rate3,:)=gather(Mint*(Mp3*reshape(f_a,n*m,N_mesh))*(2*pi));
        DDTxz(i/saving_rate3,:)=gather(Mint*(Mp1*reshape(b_DT_p3,n*m,N_mesh))*(2*pi));
        DDTzz(i/saving_rate3,:)=gather(Mint*(Mp3*reshape(b_DT_p3,n*m,N_mesh))*(2*pi));
    
        cell_den(i/saving_rate3,:)=gather(cell_den_loc);
        disp([num2str(i) '/' num2str(nsteps)]);
    end 
    if ( mod(i, saving_rate3) == (saving_rate3-2) )
        ucoeff_previous(:,:,1)=ucoeff;
    end 
    if ( mod(i, saving_rate3) == (saving_rate3-1) )
        ucoeff_previous(:,:,2)=ucoeff;
    end 
    if ( mod(i, saving_rate3) == 1 ) && i~=1
        ucoeff_previous(:,:,3)=ucoeff;
    end 
    if ( mod(i, saving_rate3) == 2 ) && i~=2
        unsteady_RHS=((-ucoeff./(real(Mint*ucoeff*2*pi))...
            + ucoeff_previous(:,:,1)./(real(Mint*ucoeff_previous(:,:,1)*2*pi)))/12 ...
            +(ucoeff_previous(:,:,3)./(real(Mint*ucoeff_previous(:,:,3)*2*pi))...
            -ucoeff_previous(:,:,2)./(real(Mint*ucoeff_previous(:,:,2)*2*pi)))*(2/3))/dt;
        
        temp=sum(bsxfun(@times,Linv,reshape([unsteady_RHS;zeros(1,N_mesh,'gpuArray')],1,n*m+1,N_mesh)),2);
        f_unsteady=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        
        Vux((i-2)/saving_rate3,:)=gather(Mint*Mp1*(reshape(f_unsteady,n*m,N_mesh))*(2*pi));
        Vuz((i-2)/saving_rate3,:)=gather(Mint*Mp3*(reshape(f_unsteady,n*m,N_mesh))*(2*pi));

    end 
end


%% Surface Integral Conservation check
t1=dt*saving_rate1:dt*saving_rate1:tfinal;
t2=dt*saving_rate2:dt*saving_rate2:tfinal;
t3=dt*saving_rate3:dt*saving_rate3:tfinal;

Nint=sum(cell_den,2)*dz;

S_profile=gather(S_profile);
Kp=gather(Kp);
ucoeff=gather(ucoeff);
ex_file_name=['smol_pBC_HS_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsmin) 'Vsm_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_cospi_cd' num2str(N_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];
ex_file_name=replace(ex_file_name,'.','-');

save([ex_file_name 'GPU.mat'],...
    'n','m','N_mesh','nsteps','S_profile','Vc','Pef','omg','beta','diff_const','DT','B','Vsmin','Vsvar',...
    'dt','tfinal','settings','Kp','z','dz',... % 'cheb',... 'dx'
    'saving_rate1','saving_rate2','saving_rate3',...
    't1','t2','t3','Nint','cell_den',...
    'ufull_save','u_zloc_save','z_sav_location','ucoeff','ucoeff0',...
    'Dxx','Dxz','Dzx','Dzz','Vix','Viz','Vux','Vuz','ex','ez','DDTxz','DDTzz','VDTx','VDTz',...
    'fdt_full_save','fndt_full_save','-v7.3');

% exit
