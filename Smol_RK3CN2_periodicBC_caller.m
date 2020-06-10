%% Full Smoluchowski Time Stepping solver for the Fokker-Planck Equation (RK3-CN2)
% Loosely based on Spherefun, this solve uses Double Fourier Sphere (DFS)
% method to tranform the orientational space and time-marched in the DFS 
% space. Time stepping is semi-implicit, with Advection terms in RK3 and 
% Laplacian term in CN2. Implicit matrix inversion is done using the
% Spherefun Helmholtz Solver (only slightly modified to remove
% chebfun-based activiities). 
parpool(20);
clear all;
%% Setting up
% Parameters
dt = 0.02;                  % Time step
tfinal = 50;                % Stopping time
nsteps = ceil(tfinal/dt);   % Number of time steps
m = 32;                     % Spatial discretization - phi (even)
n = 32;                     % Spaptial discretization - theta (even)
N_mesh=200;                 % Spaptial discretization - y
diff_const = 1;             % Diffusion constant
beta=2.2;                   % Gyrotactic time scale
% S=2.5;                      % Shear time scale
Vc=1/2.12;                       % Swimming Speed (scaled by channel width and Dr)
omg=[0,1,0];                % Vorticity direction (1,2,3) 

%Saving to settings struct
% settings.S=S;
settings.beta=beta;
settings.n=n;
settings.m=m;
settings.omg1=omg(1);
settings.omg2=omg(2);
settings.omg3=omg(3);

dx=2/(N_mesh);
x=-1:dx:1-dx;

%% Initial Condition
settings.int_const=1.;
ucoeff=zeros(n*m,N_mesh);ucoeff(m*n/2+m/2+1,:)=1/8/pi;
%% RK3 coeff and constants
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];

% Preparing Constants
K2 = (1/(dt*diff_const));         % Helmholtz frequency for BDF1
omg2_profile=-pi*sin(pi*x)*Vc*5;omg2_profile(1)=0;
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
settings.Mint=kron(fac,[zeros(1,m/2) 1 zeros(1,m/2-1)]);
settings.MintSq=settings.Mint*settings.Mint';

% Advection
% Madv=adv_mat(settings);
Mvor=adv_vor_mat(settings);
Mgyro=settings.beta*adv_gyro_mat(settings);

%Laplacian
Mlap=lap_mat(settings);
helm=helmholtz_gen( n, m);

%Dx
Rdx=spdiags(ones(N_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],N_mesh,N_mesh);
Rdx=spdiags(ones(N_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-N_mesh+3:-1:-N_mesh+1 N_mesh-1:-1:N_mesh-3],Rdx);
Rdx=Rdx/dx;

%Mp1
Mp1=kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));
%% Initialise Recorded values
Sf=NaN(nsteps,N_mesh);
% Sf_rhs1=NaN(nsteps,1);Sf_rhs2=NaN(nsteps,1);Sf_rhs3=NaN(nsteps,1);
% Sf_lap1=NaN(nsteps,1);Sf_lap2=NaN(nsteps,1);Sf_lap3=NaN(nsteps,1);
% Sf_adv1=NaN(nsteps,1);Sf_adv2=NaN(nsteps,1);Sf_adv3=NaN(nsteps,1);
% Sf(1)=real(settings.Mint*ucoeff*2*pi);

adv_p_coeff   =zeros(n*m,N_mesh);
adv_comb_coeff=zeros(n*m,N_mesh);
%% Loop!
for i = 2:nsteps
    %% RK step 1
    k=1;
    dxu_coeff=ucoeff*Rdx;
    parfor j=1:N_mesh
        settings_loc=settings;
        settings_loc.S=omg2_profile(j);
        adv_coeff=(settings_loc.S/2)*(Mvor*ucoeff(:,j))+Mgyro*ucoeff(:,j);
        adv_coeff=adv_coeff-settings_loc.Mint'*(settings_loc.Mint*adv_coeff)/settings_loc.MintSq;
        
        lap_coeff=Mlap*ucoeff(:,j);
        lap_coeff=lap_coeff-settings_loc.Mint'*(settings_loc.Mint*lap_coeff)/settings_loc.MintSq;
        
        swim_coeff=Vc*Mp1*dxu_coeff(:,j);
        
        adv_p_coeff(:,j)=adv_coeff+swim_coeff;
        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_p_coeff(:,j)));
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
    end
    %% RK step 2
    k=2;
    dxu_coeff=ucoeff*Rdx;
    parfor j=1:N_mesh
        settings_loc=settings;
        settings_loc.S=omg2_profile(j);
%         adv_p_coeff=adv_coeff+swim_coeff;
        adv_coeff=(settings_loc.S/2)*(Mvor*ucoeff(:,j))+Mgyro*ucoeff(:,j);
        adv_coeff=adv_coeff-settings_loc.Mint'*(settings_loc.Mint*adv_coeff)/settings_loc.MintSq;
        
        lap_coeff=Mlap*ucoeff(:,j);
        lap_coeff=lap_coeff-settings_loc.Mint'*(settings_loc.Mint*lap_coeff)/settings_loc.MintSq;
        
        swim_coeff=Vc*Mp1*dxu_coeff(:,j);
        adv_comb_coeff(:,j)=adv_coeff+swim_coeff;
        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_comb_coeff(:,j))+rho(k)*adv_p_coeff(:,j)); %#ok<*PFBNS>
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
    end
    %% RK step 3
    k=3;
    dxu_coeff=ucoeff*Rdx;
    adv_p_coeff=adv_comb_coeff;
    parfor j=1:N_mesh
        settings_loc=settings;
        settings_loc.S=omg2_profile(j);
        
        adv_coeff=(settings_loc.S/2)*(Mvor*ucoeff(:,j))+Mgyro*ucoeff(:,j);
        adv_coeff=adv_coeff-settings_loc.Mint'*(settings_loc.Mint*adv_coeff)/settings_loc.MintSq;
        
        lap_coeff=Mlap*ucoeff(:,j);
        lap_coeff=lap_coeff-settings_loc.Mint'*(settings_loc.Mint*lap_coeff)/settings_loc.MintSq;
        
        swim_coeff=Vc*Mp1*dxu_coeff(:,j);
        
        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_coeff+swim_coeff)+rho(k)*adv_p_coeff(:,j));
        %     % Integral Compensation
        %     rhs_coeff = rhs_coeff-settings.Mint'*(settings.Mint*rhs_coeff-settings.int_const/2/pi*(-K2/alpha(k)))/settings.MintSq;
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
        
        Sf(i,j)=real(settings_loc.Mint*ucoeff(:,j)*2*pi);
    end
    %    Plot the solution every 25 time steps
    %     if ( mod(i, 5) == 0 )
    %         u=spherefun.coeffs2spherefun(transpose(reshape(ucoeff,m,n)));
    %         contour(u,[0:0.025:0.4]);
    %         title(sprintf('Time %2.5f',i*dt)), snapnow
    %     end
%     Sf(i)=real(settings.Mint*ucoeff*2*pi);
    
end


%% Surface Integral Conservation check
% figure;
% plot([dt:dt:tfinal],Sf);
Nint=sum(Sf,2)*dx;
t=dt:dt:tfinal;
% figure;plot(t,Nint);
% figure;plot(x,Sf);

save smol_periodicBC_5Vc.mat;
exit
%% Translate back to Sphere for Post-Processing
% u=spherefun.coeffs2spherefun(transpose(reshape(ucoeff,m,n)));
% 
% n_phi=32; % Has to be even for FFT. 2^N recommended
% n_theta=101; % Had better to be 1+(multiples of 5,4,3 or 2) for Newton-Cot
% 
% dtheta=(pi/(n_theta-1));
% dphi=2*pi/(n_phi);
% 
% theta=(0:dtheta:pi)';
% phi=0:dphi:(2*pi-dphi);
% 
% 
% figure;
% contour(phi,theta,u(phi,theta));



