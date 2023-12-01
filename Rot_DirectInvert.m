%% Direct inversion for f(p)

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
Vc=0.11;                   % Swimming Speed (scaled by channel width and Dr) (Pe_s)
inv_nu=1;                % h*^2 dr^* / (kinematic viscosity=mu)
Vs=0.0;                   % Sedimentation speed scaling (delta rho g / mu *(2/9) * b^2 / (h^*dr^*), b=semi-minor

diff_const = 1.0;            % Rotational Diffusion constant (keep it at 1, for dimensional runs only)
DT=0.;                     % Translational Diffusion constant
beta=0;                    % Gyrotactic time scale
AR=5;                     % Aspect Ratio of swimmer (1=spherical) % AR=1.3778790674938353091971374518539773339097820167847;
[B,Vmin,Vmax,~,A1,A2,A3]=ellipsoid(AR);
% B=0.31;                    % Bretherton Constant of swimmer (a.k.a. alpha0, direct)
M=0;
Vsmin=Vs*Vmin;              % Minimum sedimentaion (Vs)
Vsmax=Vs*Vmax;              % Vs_max-Vs_min

h=4;
H=5;

% Discretisation
m = 16;                     % Spherical discretization - phi (even)
n = 128;                     % Spherical discretization - theta (even)

%% Flow Config
% Vertical Shear (VS)
% G= [0 0 1; ...
%     0 0 0; ...
%     0 0 0];
% Horizontal Shear (HS)
G= [0 0 0; ...
    0 0 0; ...
    1 0 0];

S_profile=5;
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
% settings.dt=dt;
% settings.d_spatial=dx;
% settings.d_spatial=dz;
% settings.N_mesh=N_mesh;
% settings.Kp=Kp;
% settings.nsteps=nsteps;

settings.omg1=G(2,3)-G(3,2);
settings.omg2=G(3,1)-G(1,3);
settings.omg3=G(1,2)-G(2,1);
settings.e11=G(1,1);
settings.e12=G(1,2)+G(2,1);
settings.e13=G(3,1)+G(1,3);
settings.e22=G(2,2);
settings.e23=G(2,3)+G(3,2);
settings.e33=G(3,3);

settings.h=h;
settings.H=H;

settings.A1=A1;
settings.A2=A2;
settings.A3=A3;
settings.AR=AR;

%%
% Advection
Mvor=adv_vor_mat(settings)+settings.B*adv_strain_mat(settings);
% Laplacian
Mlap=lap_mat(settings);
% Reflection
MReflect = adv_TwoWall1stReflect_mat(settings);

% Mgyro=settings.beta*adv_gyro_mat(settings);

% Numerical Preps
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
Msin=kron(spdiags(0.5i*ones(settings.n,1)*[-1,1], [-1 1], settings.n, settings.n),speye(settings.m));

j=1;
% Le = full(Msin*gather(S_profile(j)*(Mvor)+Mgyro-Mlap));
Le = full(gather(S_profile(j)*(Mvor+MReflect)-Mlap));
% Le = full(gather(S_profile(j)*(Mvor)-Mlap));
% Le = full(gather(S_profile(j)*(MReflect)-Mlap));
Le(n*m/2+m/2+1,:)=Mint;

g=Le\[zeros(n*m/2+m/2,1);1/2/pi;zeros(n*m/2-m/2-1,1)];

% N_mesh=numel(S_profile);
% nm=n*m;
% g=NaN(nm,N_mesh);
% Linv=NaN(nm,nm,N_mesh);
% for j=1:N_mesh
%     Le=full(Msin*gather(S_profile(j)*Mvor+Mgyro-Mlap));
%     Le(n*m/2+m/2+1,:)=Mint;
% 
%     Linv(:,:,j)=inv(Le);
% 
%     g(:,j)=Linv(:,:,j)*[zeros(n*m/2+m/2,1);1/2/pi;zeros(n*m/2-m/2-1,1)];
% end


%%
% settings.Kp=0.001;
% helm=helmholtz_genGPU( settings.n, settings.m);
% helm.dt=0.0001;
% 
% g_relax=time_relaxed_Linv(gpuArray(Mvor+MReflect),zeros(n*m,n*m,'gpuArray'),gpuArray(Mlap),gpuArray(S_profile),zeros(n*m,length(S_profile),'gpuArray'),gpuArray(Mint),gpuArray(Mint*Mint'),helm);
%%
u=spherefun.coeffs2spherefun(transpose(reshape(g,m,n)));
% u_relax=spherefun.coeffs2spherefun(transpose(reshape(g_relax,m,n)));

n_phi=32; % Has to be even for FFT. 2^N recommended
n_theta=101; % Had better to be 1+(multiples of 5,4,3 or 2) for Newton-Cot

 
dtheta=(pi/(n_theta-1));
dphi=2*pi/(n_phi);

theta=(0:dtheta:pi)';
phi=-pi:dphi:(pi-dphi);

% plot(u);
figure;
subplot(1,2,1);
contour(phi/pi,theta/pi,u(phi,theta));%axis equal;
subplot(1,2,2);
% contour(phi/pi,theta/pi,u_relax(phi,theta));axis equal;
plot(mean(u(phi,theta),2),theta/pi);