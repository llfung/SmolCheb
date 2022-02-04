%% Parameters and Environment setting
% Caller Script for Smol_RK3CN2 solvers
gpuArray(2);
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
Vc=0;                   % Swimming Speed (scaled by channel width and Dr) (Pe_s)
Pef=0.1582086;                    % Flow Peclet Number (Pe_f)
inv_nu=6.3207673579;                % h*^2 dr^* / (kinematic viscosity)
Vs=0.079795642;                   % Sedimentation speed scaling (delta rho g / mu *(2/9) * b^2 / (h^*dr^*), b=semi-minor

Ri=5;
Re=inv_nu;

diff_const = 1;            % Rotational Diffusion constant (keep it at 1, for dimensional runs only)
DT=0.;                     % Translational Diffusion constant
beta=0;                    % Gyrotactic time scale
AR=10;                     % Aspect Ratio of swimmer (1=spherical) % AR=1.3778790674938353091971374518539773339097820167847;
[B,Vmin,Vmax,M]=ellipsoid(AR);
% B=0.31;                    % Bretherton Constant of swimmer (a.k.a. alpha0, direct)
Vsmin=Vs*Vmin;              % Minimum sedimentaion (Vs)
Vsmax=Vs*Vmax;              % Vs_max-Vs_min

% Discretisation
dt = 0.002;                  % Time step
ti = 0;
tfinal = 50+dt*2;          % Stopping time
nsteps = ceil((tfinal-ti)/dt);   % Number of time steps
m = 16;                     % Spherical discretization - phi (even)
n = 32;                     % Spherical discretization - theta (even)
N_mesh=128;                 % Spatial discretization - x/z
channel_width=2.;           % Rotational Diffusion constant (keep it at 2, for dimensional runs only)

% Run settings
saving_rate1=100;
saving_rate2=Inf;
saving_rate3=100;

% Others
int_const=1.;
Kp=0.001;

epsInit=0.;

%% Name
ex_file_name=['smolbuoy_pBC_' num2str(beta) 'beta_'  num2str(DT) 'DT_' num2str(Vc) 'Vc_' num2str(AR) 'AR_' num2str(B) 'B_' num2str(M) 'M_' num2str(Vs) 'Vs_'  num2str(Re) 'Re_' num2str(Pef) 'Pef_' num2str(Ri) 'Ri_cospi_' num2str(N_mesh) 'cd_' num2str(m) 'm_' num2str(n) 'n_' num2str(dt) 'dt_' num2str(ti) 'ti_' num2str(tfinal) 'tf' ];
% ex_file_name=['smolbuoy_pBC_HS_' num2str(beta) 'beta_'  num2str(DT) 'DT_' num2str(Vc) 'Vc_' num2str(AR) 'AR_' num2str(B) 'B_' num2str(M) 'M_' num2str(Vs) 'Vs_'  num2str(Re) 'Re_' num2str(Pef) 'Pef_' num2str(Ri) 'Ri_cospi_' num2str(N_mesh) 'cd_' num2str(m) 'm_' num2str(n) 'n_' num2str(dt) 'dt_' num2str(ti) 'ti_' num2str(tfinal) 'tf' ];
% ex_file_name=['smolbuoy_rBC_' num2str(beta) 'beta_'  num2str(DT) 'DT_' num2str(Vc) 'Vc_' num2str(AR) 'AR_' num2str(B) 'B_' num2str(M) 'M_' num2str(Vs) 'Vs_'  num2str(Re) 'Re_' num2str(Pef) 'Pef_' num2str(Ri) 'Ri_cospi_' num2str(N_mesh) 'cd_' num2str(m) 'm_' num2str(n) 'n_' num2str(dt) 'dt_' num2str(ti) 'ti_' num2str(tfinal) 'tf' ];
% ex_file_name=['smolbuoy_rBC_HS_' num2str(beta) 'beta_'  num2str(DT) 'DT_' num2str(Vc) 'Vc_' num2str(AR) 'AR_' num2str(B) 'B_' num2str(M) 'M_' num2str(Vs) 'Vs_'  num2str(Re) 'Re_' num2str(Pef) 'Pef_' num2str(Ri) 'Ri_cospi_' num2str(N_mesh) 'cd_' num2str(m) 'm_' num2str(n) 'n_' num2str(dt) 'dt_' num2str(ti) 'ti_' num2str(tfinal) 'tf' ];

%% Preliminary Meshing
dx=channel_width/(N_mesh);
x=-(channel_width/2):dx:(channel_width/2)-dx;
% dz=channel_width/(N_mesh);
% z=-(channel_width/2):dz:(channel_width/2)-dz;

% cheb=chebyshev(N_mesh,2,bc_type.none,tran_type.none);
% x=cheb.col_pt;Rdx=(cheb.D(1))';Rd2x=(cheb.D(2))';
% cheb=chebyshev(N_mesh,2,bc_type.none,tran_type.none);
% z=cheb.col_pt;Rdz=(cheb.D(1))';Rd2z=(cheb.D(2))';

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
W_profile=-(zeros(size(x))+1*cos(pi*x))*Pef;
S_profile=zeros(size(x));
%     W_profile=(-cos(pi*x)-1)*Pef;   % W(x)=-cos(pi x)-1
%     S_profile=pi*sin(pi*x)*Pef/2;     % .5*dW(x)/dx=pi*sin(pi x)/2
%     S_profile(1)=0;
% Horizontal Shear (HS)
%     % U_profile=(cos(pi*z)+1)*Pef;     % U(z)=cos(pi x)+1
%     S_profile=-pi*sin(pi*z)*Pef/2;     % .5*dU(z)/dz=-pi*sin(pi x)/2
%     S_profile(1)=0;
% Others
% S_profile=x*Pef;                    % W(x)=-(1-x^2)
% S_profile=Pef/2*ones(size(x));      % W(x)=x

W_prof0=W_profile;

%% Saving to settings struct
settings.beta=beta;
settings.B=B;
settings.Vc=Vc;
settings.inv_nu=inv_nu;
settings.M=M;
settings.Vsmin=Vsmin;
settings.Vsmax=Vsmax;
settings.n=n;
settings.m=m;
settings.diff_const=diff_const;
settings.dt=dt;
settings.d_spatial=dx;
% settings.d_spatial=dz;
settings.N_mesh=N_mesh;
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
if ti==0
    ucoeff0=zeros(n*m,N_mesh);
    ucoeff0(m*n/2+m/2+1,:)=int_const/4/pi/channel_width;
else
    load(['smolbuoy_pBC_' num2str(beta) 'beta_' num2str(AR) 'AR_' num2str(Vc) 'Vc_' num2str(Vs) 'Vs_' num2str(DT) 'DT_' num2str(Re) 'Re_' num2str(Ri) 'Ri_' num2str(Pef) 'Pef_cospi_cd' num2str(N_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_ti' num2str(0) '_tf' num2str(ti)],...
        'ucoeff');
    ucoeff0=ucoeff;
end

% norm_distr=exp(-x.^2/epsInit);
% norm_distr=norm_distr/(sum(norm_distr)*dx);
% ucoeff0(m*n/2+m/2+1,:)=(1/4/pi)*norm_distr;

%% Call Run Script
% SmolBuoy_RK3CN2_pBC;
SmolBuoy_RK3CN2_pBC_GPU;

%% Final PS
t1=(dt*saving_rate1:dt*saving_rate1:(tfinal-ti))+ti;
t2=(dt*saving_rate2:dt*saving_rate2:(tfinal-ti))+ti;
t3=(dt*saving_rate3:dt*saving_rate3:(tfinal-ti))+ti;

Nint=sum(cell_den,2)*dx;
% Nint=sum(cell_den,2)*dz;

%% Gathering Data
% g=PS.g;
% Transformed=PS.export_struct();
% [ex,ez,ex_g,ez_g,Dxx,Dxz,Dzx,Dzz,Dxx_g,Dxz_g,Dzx_g,Dzz_g,Vix,Viz,Vix_g,Viz_g,VDTx,VDTz,VDTx_g,VDTz_g,DDTxz,DDTzz,DDTxz_g,DDTzz_g]=PS.export();
% [ex,ez,ex_g,ez_g,Dxx,Dxz,Dzx,Dzz,Dxx_g,Dxz_g,Dzx_g,Dzz_g,Vix,Viz,Vix_g,Viz_g,VDTx,VDTz,VDTx_g,VDTz_g,DDTxz,DDTzz,DDTxz_g,DDTzz_g,Vux,Vuz]=PS.export();

settings.Mint=gather(settings.Mint);
S_profile=gather(S_profile);
Kp=gather(Kp);
ucoeff=gather(ucoeff);

save([ex_file_name '.mat'],...
    'Vc','Pef','AR','inv_nu','Vs','Re','Ri','B','M',...
    'diff_const','beta','DT',...
    'dt','tfinal','nsteps','m','n','N_mesh','channel_width',...
    'int_const','Kp','epsInit',...
    'x','dx',...'z','dz',...
    'S_profile','G','ucoeff0',...
    't1','t2','t3',...
    'settings','ucoeff','cell_den','Nint',...
    'ex_file_name',...
    'W_prof','W_prof0',...
...%     'g','Transformed',...
    '-v7.3');


