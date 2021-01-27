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
Vc=0.25;                   % Swimming Speed (scaled by channel width and Dr) (Pe_s)
Pef=1.;                     % Flow Peclet Number (Pe_f)
Vsmin=0.;                  % Minimum sedimentaion (Vs)
Vsvar=0.;                  % Vs_max-Vs_min

diff_const = 1;            % Rotational Diffusion constant (keep it at 1, for dimensional runs only)
DT=0.;                     % Translational Diffusion constant
beta=2.2;                  % Gyrotactic time scale
% AR=20;                   % Aspect Ratio of swimmer (1=spherical) % AR=1.3778790674938353091971374518539773339097820167847;
% B=(AR^2-1)/(AR^2+1);     % Bretherton Constant of swimmer (a.k.a. alpha0, through AR)
B=0.31;                    % Bretherton Constant of swimmer (a.k.a. alpha0, direct)

% Discretisation
dt = 0.01;                  % Time step
tfinal = 1+dt*2;           % Stopping time
nsteps = ceil(tfinal/dt);   % Number of time steps
m = 20;                     % Spherical discretization - phi (even)
n = 32;                     % Spherical discretization - theta (even)
N_mesh=128;                 % Spatial discretization - x/z
channel_width=2.;           % Rotational Diffusion constant (keep it at 2, for dimensional runs only)

% Run settings
saving_rate1=2000;
saving_rate2=50;
saving_rate3=25;

% Others
int_const=1.;
Kp=0.01;

epsInit=0.;

%% Preliminary Meshing
% dx=channel_width/(N_mesh);
% x=-(channel_width/2):dx:(channel_width/2)-dx;
dz=channel_width/(N_mesh);
z=-(channel_width/2):dz:(channel_width/2)-dz;

% cheb=chebyshev(N_mesh,2,bc_type.none,tran_type.none);
% x=cheb.col_pt;Rdx=(cheb.D(1))';Rd2x=(cheb.D(2))';
% cheb=chebyshev(N_mesh,2,bc_type.none,tran_type.none);
% z=cheb.col_pt;Rdz=(cheb.D(1))';Rd2z=(cheb.D(2))';

%% Flow Config
% Vertical Shear (VS)
% G= [0 0 1; ...
%     0 0 0; ...
%     0 0 0];
% Horizontal Shear (HS)
G= [0 0 0; ...
    0 0 0; ...
    1 0 0];

% Shear Profile
% Vertical Shear (VS)
    % W_profile=(-cos(pi*x)-1)*Pef;   % W(x)=-cos(pi x)-1
%     S_profile=pi*sin(pi*x)*Pef/2;     % .5*dW(x)/dx=pi*sin(pi x)/2
%     S_profile(1)=0;
% Horizontal Shear (HS)
%     % U_profile=(cos(pi*z)+1)*Pef;     % U(z)=cos(pi x)+1
    S_profile=-pi*sin(pi*z)*Pef/2;     % .5*dU(z)/dz=-pi*sin(pi x)/2
    S_profile(1)=0;
% Others
% S_profile=x*Pef;                    % W(x)=-(1-x^2)
% S_profile=Pef/2*ones(size(x));      % W(x)=x

%% Initial Condition
ucoeff0=zeros(n*m,N_mesh);
ucoeff0(m*n/2+m/2+1,:)=int_const/4/pi/channel_width;

% norm_distr=exp(-x.^2/epsInit);
% norm_distr=norm_distr/(sum(norm_distr)*dx);
% ucoeff0(m*n/2+m/2+1,:)=(1/4/pi)*norm_distr;

%% Call Run Script
% Smol_RK3CN2_pBC;
% Smol_RK3CN2_pBC_GPU;
% Smol_RK3CN2_pBC_HS;
Smol_RK3CN2_pBC_HS_GPU;
% Smol_RK3CN2_rBC;
% Smol_RK3CN2_rBC_HS;

%% Final PS
t1=dt*saving_rate1:dt*saving_rate1:tfinal;
t2=dt*saving_rate2:dt*saving_rate2:tfinal;
t3=dt*saving_rate3:dt*saving_rate3:tfinal;

% Nint=sum(cell_den,2)*dx;
Nint=sum(cell_den,2)*dz;

S_profile=gather(S_profile);
Kp=gather(Kp);
ucoeff=gather(ucoeff);

% ex_file_name=['smol_pBC_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsmin) 'Vsm_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_cospi_cd' num2str(N_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];
ex_file_name=['smol_pBC_HS_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsmin) 'Vsm_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_cospi_cd' num2str(N_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];
% ex_file_name=['smol_rBC_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsmin) 'Vsm_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_cospi_cd' num2str(N_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];
% ex_file_name=['smol_rBC_HS_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsmin) 'Vsm_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_cospi_cd' num2str(N_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];

save('Summary.mat',...
    'Vc','Pef','Vsmin','Vsvar',...
    'diff_const','DT','beta','B',...'AR',...
    'dt','tfinal','nsteps','m','n','N_mesh','channel_width',...
    'int_const','Kp','epsInit',...
    'z','dz',...'x','dx',...
    'S_profile','G','ucoeff0',...
    't1','t2','t3',...
    'settings','ucoeff','cell_den','Nint',...
    'ex_file_name',...
    '-v7.3');


