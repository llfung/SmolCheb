%% Parameters and Environment setting
% Caller Script for Smol_RK3CN2 solvers

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
Vc=0.025;                   % Swimming Speed (scaled by channel width and Dr) (Pe_s)
Pef=0.5;                    % Flow Peclet Number (Pe_f)
Vsmin=0.;                  % Minimum sedimentaion (Vs)
Vsvar=0.;                  % Vs_max-Vs_min

diff_const = 1;            % Rotational Diffusion constant (keep it at 1, for dimensional runs only)
DT=0.;                     % Translational Diffusion constant
beta=2.2;                  % Gyrotactic time scale
% AR=20;                   % Aspect Ratio of swimmer (1=spherical) % AR=1.3778790674938353091971374518539773339097820167847;
% B=(AR^2-1)/(AR^2+1);     % Bretherton Constant of swimmer (a.k.a. alpha0, through AR)
B=0.31;                    % Bretherton Constant of swimmer (a.k.a. alpha0, direct)

% Discretisation
dt = 0.0025;                  % Time step
tfinal = 1+dt*2;          % Stopping time
nsteps = ceil(tfinal/dt);   % Number of time steps
m = 8;                      % Spherical discretization - phi (even)
n = 12;                     % Spherical discretization - theta (even)
Nx_mesh=96;                 % Spatial discretization - x
Nz_mesh=128;                % Spectral discretization - z
x_width=2.;                 % Channel Width  (keep it at 2, for dimensional runs only)
z_width=8.;                 % Channel Height (keep it at 2, for dimensional runs only)

% Run settings
saving_rate1=100;
saving_rate2=Inf;
saving_rate3=40;

% Others
int_const=1.;
Kp=0.001;

epsInit=0.005;

%% Preliminary Meshing
dx=x_width/(Nx_mesh);
x=-(x_width/2):dx:(x_width/2)-dx;
dz=z_width/(Nz_mesh);
z=(-z_width/2:dz:z_width/2-dz);

alphaK=2*pi/z_width;

%% Flow Config
% Vertical Shear (VS)
G= [0 0 1; ...
    0 0 0; ...
    0 0 0];
% Horizontal Shear (HS)
% G= [0 0 0; ...
%     0 0 0; ...
%     1 0 0];

% Shear Profile
% Vertical Shear (VS)
W_profile=gpuArray(repmat(x*Pef,1,Nz_mesh));   % W(x)=-cos(pi x)-1
U_profile=gpuArray(zeros(1,Nx_mesh*Nz_mesh));
S_profile=gpuArray(repmat(ones(size(x))*(Pef/2),1,Nz_mesh)); 
% Horizontal Shear (HS)
% U_profile=gpuArray(reshape(repmat(z*Pef,Nx_mesh,1),1,Nx_mesh*Nz_mesh));   % W(x)=-cos(pi x)-1
% W_profile=gpuArray(zeros(1,Nx_mesh*Nz_mesh));
% S_profile=gpuArray(reshape(repmat(ones(size(z))*Pef/2,Nx_mesh,1),1,Nx_mesh*Nz_mesh)); 

%% Saving to settings struct
settings.beta=beta;
settings.B=B;
settings.Vc=Vc;
settings.n=n;
settings.m=m;
settings.diff_const=diff_const;
settings.dt=dt;
settings.d_spatial=dx;
% settings.d_spatial=dz;
settings.N_mesh=Nx_mesh;
settings.Kp=Kp;
settings.nsteps=nsteps;

settings.omg1=G(2,3)-G(3,2);
settings.omg2=G(3,1)-G(1,3);
settings.omg3=G(1,2)-G(2,1);
settings.e11=G(1,1);
settings.e12=G(1,2)+G(2,1);
settings.e13=G(3,1)+G(1,3);
settings.e22=G(2,2);
settings.e23=G(2,3)+G(3,2);
settings.e33=G(3,3);
settings.int_const=int_const;

%% Initial Condition
norm_distrx=exp(-x.^2/epsInit);
norm_distrx=norm_distrx/(sum(norm_distrx)*dx);
norm_distrz=exp(-z.^2/epsInit);
norm_distrz=norm_distrz/(sum(norm_distrz)*dz);
norm_distrz_T=fft(norm_distrz)/Nz_mesh;

%% Call Run Script
Smol_RK3CN2_xpBC_zFourier_GPU;

%% Final PS
t1=dt*saving_rate1:dt*saving_rate1:tfinal;
t2=dt*saving_rate2:dt*saving_rate2:tfinal;
t3=dt*saving_rate3:dt*saving_rate3:tfinal;

Nint=gather(Nint_loc);
int_const=gather(int_const);

%% Gathering Data
% g=PS.g;
% Transformed=PS.export_struct();
% [ex,ez,ex_g,ez_g,Dxx,Dxz,Dzx,Dzz,Dxx_g,Dxz_g,Dzx_g,Dzz_g,Vix,Viz,Vix_g,Viz_g,VDTx,VDTz,VDTx_g,VDTz_g,DDTxz,DDTzz,DDTxz_g,DDTzz_g]=PS.export();
% [ex,ez,ex_g,ez_g,Dxx,Dxz,Dzx,Dzz,Dxx_g,Dxz_g,Dzx_g,Dzz_g,Vix,Viz,Vix_g,Viz_g,VDTx,VDTz,VDTx_g,VDTz_g,DDTxz,DDTzz,DDTxz_g,DDTzz_g,Vux,Vuz]=PS.export();

settings.Mint=gather(settings.Mint);
S_profile=gather(S_profile);
Kp=gather(Kp);
ucoeff=gather(ucoeff);

ex_file_name=['smol_pBC_2D_' num2str(epsInit) 'epsInit_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsmin) 'Vsm_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_homo_dx_' num2str(Nx_mesh) 'dz_' num2str(Nz_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];
% ex_file_name=['smol_pBC_HS_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsmin) 'Vsm_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_cospi_cd' num2str(N_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];
% ex_file_name=['smol_rBC_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsmin) 'Vsm_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_cospi_cd' num2str(N_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];
% ex_file_name=['smol_rBC_HS_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsmin) 'Vsm_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_cospi_cd' num2str(N_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];

save('Summary.mat',...
    'Vc','Pef','Vsmin','Vsvar',...
    'diff_const','DT','beta','B',...'AR',...
    'dt','tfinal','nsteps','m','n','Nx_mesh','Nz_mesh','x_width','z_width',...
    'int_const','Kp','epsInit',...
    'x','dx','z','dz',...
    'S_profile','G','ucoeff0',...
    't1','t2','t3',...
    'settings','ucoeff','cell_den','Nint',...
    'ex_file_name',...
...%    'g','Transformed',...
    '-v7.3');


