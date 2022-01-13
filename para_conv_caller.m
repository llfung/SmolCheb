%% Parameters and Environment setting
% Caller Script for Smol_RK3CN2 solvers
%gpuDevice(2);
%% Environmental set up
addpath(genpath('core'))
addpath(genpath('core/x_FD'))
addpath(genpath('core/p_DFS'))
addpath(genpath('core/PS_RT'))
% Parallelisation
% For CX1
    % pc=parcluster('local');
    % pc.JobStorageLocation = strcat(getenv('TMPDIR'),'/para_tmp');
    % par=parpool(pc,32);
% For 1st/2nd Machine
    % parpool(20);
clear all;

%% Parameters
% Numericals
Vc=0.25;                   % Swimming Speed (scaled by channel width and Dr) (Pe_s)
Pef=1;                    % Flow Peclet Number (Pe_f)

diff_const = 1;            % Rotational Diffusion constant (keep it at 1, for dimensional runs only)
DT=0.000;                     % Translational Diffusion constant
beta=2.2;                  % Gyrotactic time scale
% AR=20;                   % Aspect Ratio of swimmer (1=spherical) % AR=1.3778790674938353091971374518539773339097820167847;
% B=(AR^2-1)/(AR^2+1);     % Bretherton Constant of swimmer (a.k.a. alpha0, through AR)
B=0.31;                    % Bretherton Constant of swimmer (a.k.a. alpha0, direct)

% Discretisation
dt = 0.00005;                  % Time step
ti= 13.5;
tfinal = 15;          % Stopping time
nsteps = ceil((tfinal-ti)/dt);   % Number of time steps
m = 8;                      % Spherical discretization - phi (even)
n = 10;                     % Spherical discretization - theta (even)
Nx_mesh=450;                 % Spatial discretization - x
Nz_mesh=450;                % Spectral discretization - z
x_width=2.;                 % Channel Width  (keep it at 2, for dimensional runs only)
z_width=2.;                 % Channel Height (keep it at 2, for dimensional runs only)

% Run settings
saving_rate1=5000;
saving_rate2=Inf;
saving_rate3=5000;

% Others
int_const=1.;
Kp=0.000001;

%% Preliminary Meshing
dx=x_width/(Nx_mesh);
x=-(x_width/2):dx:(x_width/2)-dx;
dz=z_width/(Nz_mesh);
z=(-z_width/2:dz:z_width/2-dz);

%% Flow Config
% Velocity Profile (Bearon 2011)
   U_profile=gpuArray(reshape(sin(pi*x')*cos(pi*z)*Pef,1,Nx_mesh*Nz_mesh));
   W_profile=gpuArray(reshape(-cos(pi*x')*sin(pi*z)*Pef,1,Nx_mesh*Nz_mesh));
curl_profile=gpuArray(reshape(sin(pi*x')*sin(pi*z)*Pef*pi*2,1,Nx_mesh*Nz_mesh));
   E_profile=gpuArray(reshape(cos(pi*x')*cos(pi*z)*Pef*pi,1,Nx_mesh*Nz_mesh));

%% Saving to settings struct
settings.beta=beta;
settings.B=B;
settings.Vc=Vc;
settings.n=n;
settings.m=m;
settings.diff_const=diff_const;
settings.dt=dt;
%settings.d_spatial=dx;
%settings.d_spatial=dz;
%settings.N_mesh=Nx_mesh;
settings.Kp=Kp;
settings.nsteps=nsteps;

settings.omg1=0;
settings.omg2=-1;
settings.omg3=0;
settings.e11=1;
settings.e12=0;
settings.e13=0;
settings.e22=0;
settings.e23=0;
settings.e33=-1;
settings.int_const=int_const;

%% Setting Matrices
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
Mint=kron(fac,[zeros(1,m/2) 1 zeros(1,m/2-1)]);
MintSq=Mint*Mint';
settings.Mint=Mint;
settings.MintSq=MintSq;

settings.Kp=settings.Kp/MintSq/settings.diff_const/settings.dt;

% Advection
Mvor=adv_vor_mat(settings);
Mstrain=settings.B*adv_strain_mat(settings);
Mgyro=settings.beta*adv_gyro_mat(settings);

%Laplacian
Mlap=lap_mat(settings);

Rd=spdiags(ones(Nx_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],Nx_mesh,Nx_mesh);
Rd=spdiags(ones(Nx_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-Nx_mesh+3:-1:-Nx_mesh+1 Nx_mesh-1:-1:Nx_mesh-3],Rd);
Rd=Rd/dx;
Rd2=spdiags(ones(Nx_mesh,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90],[3:-1:-3],Nx_mesh,Nx_mesh);
Rd2=spdiags(ones(Nx_mesh,1)*[1/90 -3/20 3/2 3/2 -3/20 1/90],[-Nx_mesh+3:-1:-Nx_mesh+1 Nx_mesh-1:-1:Nx_mesh-3],Rd2);
Rd2=Rd2/dx/dx;
Rdx=gpuArray(kron(speye(Nz_mesh),Rd));
Rd2x=gpuArray(kron(speye(Nz_mesh),Rd2));

Rd=spdiags(ones(Nz_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],Nz_mesh,Nz_mesh);
Rd=spdiags(ones(Nz_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-Nz_mesh+3:-1:-Nz_mesh+1 Nz_mesh-1:-1:Nz_mesh-3],Rd);
Rd=Rd/dz;
Rd2=spdiags(ones(Nz_mesh,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90],[3:-1:-3],Nz_mesh,Nz_mesh);
Rd2=spdiags(ones(Nz_mesh,1)*[1/90 -3/20 3/2 3/2 -3/20 1/90],[-Nz_mesh+3:-1:-Nz_mesh+1 Nz_mesh-1:-1:Nz_mesh-3],Rd2);
Rd2=Rd2/dz/dz;
Rdz=gpuArray(kron(Rd,speye(Nz_mesh)));
Rd2z=gpuArray(kron(Rd2,speye(Nz_mesh)));

%p1
Mp1 = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));
%p3
Mp3 = kron(spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n),speye(m));

%% Call Run Script
Smol_RK3CN2_xpBC_zpBC_GPU;

%% Final PS
t1=(dt*saving_rate1:dt*saving_rate1:tfinal)+ti;
t2=dt*saving_rate2:dt*saving_rate2:tfinal;
t3=(dt*saving_rate3:dt*saving_rate3:tfinal)+ti;

Nint=gather(Nint_loc);
int_const=gather(int_const);

%% Gathering Data
% g=PS.g;
% Transformed=PS.export_struct();
% [ex,ez,ex_g,ez_g,Dxx,Dxz,Dzx,Dzz,Dxx_g,Dxz_g,Dzx_g,Dzz_g,Vix,Viz,Vix_g,Viz_g,VDTx,VDTz,VDTx_g,VDTz_g,DDTxz,DDTzz,DDTxz_g,DDTzz_g]=PS.export();
% [ex,ez,ex_g,ez_g,Dxx,Dxz,Dzx,Dzz,Dxx_g,Dxz_g,Dzx_g,Dzz_g,Vix,Viz,Vix_g,Viz_g,VDTx,VDTz,VDTx_g,VDTz_g,DDTxz,DDTzz,DDTxz_g,DDTzz_g,Vux,Vuz]=PS.export();

settings.Mint=gather(settings.Mint);
U_profile=gather(U_profile);
W_profile=gather(W_profile);
curl_profile=gather(curl_profile);
E_profile=gather(E_profile);
Kp=gather(Kp);
ucoeff=gather(ucoeff);

ex_file_name=['smol_pBC_bearon2011_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vc) 'Vc_' num2str(DT) 'DT_'  num2str(Pef) 'Pef_dx_' num2str(Nx_mesh) 'dz_' num2str(Nz_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_ti' num2str(ti) '_tf' num2str(tfinal)];

save([ex_file_name '.mat'],...
    'Vc','Pef',...
    'diff_const','beta','B','DT',...
    'dt','ti','tfinal','nsteps','m','n','Nx_mesh','Nz_mesh','x_width','z_width',...
    'int_const','Kp',...
    'x','dx','z','dz',...
    'U_profile','W_profile','curl_profile','E_profile',...
    'ucoeff0',...
    't1','t2','t3',...
    'settings','ucoeff','cell_den','Nint',...
    'ex_file_name',...
...%    'g','Transformed',...
    '-v7.3');

