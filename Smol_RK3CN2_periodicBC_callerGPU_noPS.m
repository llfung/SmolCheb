%% Full Smoluchowski Time Stepping solver for the Fokker-Planck Equation (RK3-CN2)
% Loosely based on Spherefun, this solve uses Double Fourier Sphere (DFS)
% method to tranform the orientational space and time-marched in the DFS 
% space. Time stepping is semi-implicit, with Advection terms in RK3 and 
% Laplacian term in CN2. Implicit matrix inversion is done using the
% Spherefun Helmholtz Solver (only slightly modified to remove
% chebfun-based activiities). 

%% Setting up
% Parameters
Vc=0.25;                       % Swimming Speed (scaled by channel width and Dr) (Pe_s)
Pef=1;                      % Flow Peclet Number (Pe_f)
% Vsmin=0.2;                  % Minimum sedimentaion (Vs)
Vsvar=0;                  % Vs_max-Vs_min

diff_const = 1;             % Rotational Diffusion constant
DT=.0;                      % Translational Diffusion constant
beta=2.2;                   % Gyrotactic time scale
% AR=1;                      % Aspect Ratio of swimmer (1=spherical) % AR=1.3778790674938353091971374518539773339097820167847;
% B=(AR^2-1)/(AR^2+1);        % Bretherton Constant of swimmer (a.k.a. alpha0)
B=0.31;

dt = 0.0025;                  % Time step
tfinal = 2+dt*2;           % Stopping time
nsteps = ceil(tfinal/dt);   % Number of time steps
m = 20;                     % Spatial discretization - phi (even)
n = 32;                     % Spaptial discretization - theta (even)
N_mesh=256;                 % Spaptial discretization - x

omg=[0,-1,0];               % Vorticity direction (1,2,3) 

% Run saving settings
% saving_rate1=1000000;
saving_rate2=40;
saving_rate3=40;

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
dx=2/(N_mesh);
x=-1:dx:1-dx;
% cheb=chebyshev(N_mesh,2,bc_type.none,tran_type.none);
% z=cheb.col_pt;
% D1=(cheb.D(1))';D2=(cheb.D(2))';

%% Shear Profile
% W_profile=(-cos(pi*x)-1)*Pef;   % W(x)=-cos(pi x)-1
S_profile= pi*sin(pi*x)*Pef/2; % .5*dW(x)/dx= pi*sin(pi x)/2
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
Rdx=spdiags(ones(N_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],N_mesh,N_mesh);
Rdx=spdiags(ones(N_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-N_mesh+3:-1:-N_mesh+1 N_mesh-1:-1:N_mesh-3],Rdx);
Rdx=gpuArray(Rdx/dx);
Rd2x=spdiags(ones(N_mesh,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90],[3:-1:-3],N_mesh,N_mesh);
Rd2x=spdiags(ones(N_mesh,1)*[1/90 -3/20 3/2 3/2 -3/20 1/90],[-N_mesh+3:-1:-N_mesh+1 N_mesh-1:-1:N_mesh-3],Rd2x);
Rd2x=gpuArray(Rd2x/dx/dx);

%p1
Mp1 = gpuArray(complex(kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m))));
%p3
% Mp3 = gpuArray(sparse(complex(full(kron(spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n),speye(m)))))); %e3
%p1p3
Mp1p3 = kron(spdiags(.25i*ones(n,1)*[-1,1], [-2 2], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));

%Swimming and sedimentation   
MSwim=Vc*Mp1-Vsvar*Mp1p3;

%% Initial Condition
int_const=1.;
settings.int_const=int_const;

ucoeff0=zeros(n*m,N_mesh);
ucoeff0(m*n/2+m/2+1,:)=1/8/pi;

%% Initialise Recorded values
cell_den=NaN(floor(nsteps/saving_rate3),N_mesh);

ufull_save=NaN(n*m,N_mesh);
% fdt_full_save=NaN(n*m,N_mesh);
% fndt_full_save=NaN(n*m,N_mesh);

%% Time-Stepping (RK3-CN2)
ucoeff=gpuArray(complex(ucoeff0));
% adv_p_coeff   =gpuArray(complex(zeros(n*m,N_mesh)));
% adv_comb_coeff=gpuArray(complex(zeros(n*m,N_mesh)));
ucoeff_previous=gpuArray(complex(NaN(n*m,N_mesh,3)));

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dx;

for i = 1:nsteps
    %% RK step 1
    k=1;
    % Par-For Version
    dxu_coeff=ucoeff*Rdx;
    dx2u_coeff=ucoeff*Rd2x;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    swim_coeff=MSwim*dxu_coeff;
    DT_coeff=DT*dx2u_coeff;
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
    Nint_loc=sum(cell_den_loc,2)*dx;
    
    %% RK step 2
    k=2;
    dxu_coeff=ucoeff*Rdx;
    dx2u_coeff=ucoeff*Rd2x;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    swim_coeff=MSwim*dxu_coeff;
    DT_coeff=DT*dx2u_coeff;
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
    Nint_loc=sum(cell_den_loc,2)*dx;
    
    %% RK step 3
    k=3;
    dxu_coeff=ucoeff*Rdx;
    dx2u_coeff=ucoeff*Rd2x;
    adv_p_coeff=adv_comb_coeff;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    swim_coeff=MSwim*dxu_coeff;
    DT_coeff=DT*dx2u_coeff;
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
    Nint_loc=sum(cell_den_loc,2)*dx;
    
    %% Saving for Post-Processing
    %    Plot/Save the solution every saving_rate
    if ( mod(i, saving_rate2) == 0 )
        ufull_save=gather(ucoeff);
    end
    if ( mod(i, saving_rate2) == 2 )&& i~=2 
        fdt_full_save=gather((-ucoeff./(real(Mint*ucoeff*2*pi))...
            + ucoeff_previous(:,:,1)./(real(Mint*ucoeff_previous(:,:,1)*2*pi)))/12 ...
            +(ucoeff_previous(:,:,3)./(real(Mint*ucoeff_previous(:,:,3)*2*pi))...
            -ucoeff_previous(:,:,2)./(real(Mint*ucoeff_previous(:,:,2)*2*pi)))*(2/3))/dt;
        fndt_full_save=gather((-ucoeff...
            + ucoeff_previous(:,:,1))/12 ...
            +(ucoeff_previous(:,:,3)...
            -ucoeff_previous(:,:,2))*(2/3))/dt;
        
        t=(i-2)*dt;
        save(['smol_pBC_t' num2str(t) '.mat'],'t','ufull_save','fdt_full_save','fndt_full_save');
    end
    if ( mod(i, saving_rate2) == 1 )&& i~=1 
        ucoeff_previous(:,:,3)=ucoeff;
    end
    if ( mod(i, saving_rate2) == saving_rate2-1 )
        ucoeff_previous(:,:,2)=ucoeff;
    end
    if ( mod(i, saving_rate2) == saving_rate2-2 )
        ucoeff_previous(:,:,1)=ucoeff;
    end
    
    %% On-the-go-Post-Processing
     if ( mod(i, saving_rate3) == 0 )
        cell_den(i/saving_rate3,:)=gather(cell_den_loc);
        disp([num2str(i) '/' num2str(nsteps)]);
     end

end


%% Surface Integral Conservation check
t2=dt*saving_rate2:dt*saving_rate2:tfinal;
t3=dt*saving_rate3:dt*saving_rate3:tfinal;

Nint=sum(cell_den,2)*dx;

S_profile=gather(S_profile);
Kp=gather(Kp);
ucoeff=gather(ucoeff);
ex_file_name=['smol_pBC_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_cospi_cd' num2str(N_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];
ex_file_name=replace(ex_file_name,'.','-');

save([ex_file_name 'GPU.mat'],...
    'n','m','N_mesh','nsteps','S_profile','Vc','Pef','omg','beta','diff_const','DT','B','Vsvar',...
    'dt','tfinal','settings','Kp','x','dx',... % 'cheb',... 'dx'
    'saving_rate2','saving_rate3',...
    't2','t3','Nint','cell_den',...
    'ucoeff','ucoeff0',...
    '-v7.3');

exit
