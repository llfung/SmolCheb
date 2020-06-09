clear all;

%% Setting up
dt = 0.05;                         % Time step
tfinal = 16;                        % Stopping time
nsteps = ceil(tfinal/dt);          % Number of time steps
m = 32;                            % Spatial discretization - phi
n = 32;                            % Spaptial discretization - theta
diff_const = 1;                      % Diffusion constant
beta=2.2;
S=2.5;
settings.S=S;
settings.beta=beta;
settings.n=n;
settings.m=m;
settings.omg1=0;
settings.omg2=-1;
settings.omg3=0;
%% Integral weight

arr=[-n/2:n/2-1];
fac=2./(1-arr.^2);
if mod(n/2,2)
    fac(1:2:end)=0;
    fac(n/2)=0;
    fac(n/2+2)=0;
else
    fac(2:2:end)=0;
end



%% Initial Condition
u0 = spherefun.sphharm(0,0)/sqrt(4*pi)+(spherefun.sphharm(6,0) + sqrt(14/11)*spherefun.sphharm(6,5))/sqrt(4*pi)/3;
% u0 = spherefun.sphharm(6,0) + sqrt(14/11)*spherefun.sphharm(6,5);
ucoeff=reshape(transpose(coeffs2(u0,m,n)),n*m,1);
settings.int_const=1.;
% ucoeff=zeros(n*m,1);ucoeff(m*n/2+m/2+1,1)=1/4/pi;
%% RK3 coeff and constants
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];


% Preparing Constants
K2 = (1/(dt*diff_const));         % Helmholtz frequency for BDF1

%% Initialising Matrices
% Surface Integrals
settings.Mint=kron(fac,[zeros(1,m/2) 1 zeros(1,m/2-1)]);
settings.MintSq=settings.Mint*settings.Mint';

% Advection
Madv=adv_mat(settings);

%Laplacian
Mlap=lap_mat(settings);

%% Initialise Recorded values
Sf=NaN(nsteps,1);
% Sf_rhs1=NaN(nsteps,1);Sf_rhs2=NaN(nsteps,1);Sf_rhs3=NaN(nsteps,1);
% Sf_lap1=NaN(nsteps,1);Sf_lap2=NaN(nsteps,1);Sf_lap3=NaN(nsteps,1);
% Sf_adv1=NaN(nsteps,1);Sf_adv2=NaN(nsteps,1);Sf_adv3=NaN(nsteps,1);
Sf(1)=real(settings.Mint*ucoeff*2*pi);
%% Loop!
for i = 2:nsteps
    
    k=1;

    adv_coeff=Madv*ucoeff;
    adv_coeff=adv_coeff-settings.Mint'*(settings.Mint*adv_coeff)/settings.MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-settings.Mint'*(settings.Mint*lap_coeff)/settings.MintSq;

    rhs_coeff = -K2/alpha(k)*ucoeff-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*adv_coeff);
    ucoeff = helmholtz_coeff(rhs_coeff, -K2/alpha(k), n, m);
    
    k=2;
    
    adv_p_coeff=adv_coeff;
    adv_coeff=Madv*ucoeff;
    adv_coeff=adv_coeff-settings.Mint'*(settings.Mint*adv_coeff)/settings.MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-settings.Mint'*(settings.Mint*lap_coeff)/settings.MintSq;
    
    rhs_coeff = -K2/alpha(k)*ucoeff-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*adv_coeff+rho(k)*adv_p_coeff);
    ucoeff = helmholtz_coeff(rhs_coeff, -K2/alpha(k), n, m);    
    
    k=3;
    
    adv_p_coeff=adv_coeff;
    adv_coeff=Madv*ucoeff;
    adv_coeff=adv_coeff-settings.Mint'*(settings.Mint*adv_coeff)/settings.MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-settings.Mint'*(settings.Mint*lap_coeff)/settings.MintSq;
    
    rhs_coeff = -K2/alpha(k)*ucoeff-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*adv_coeff+rho(k)*adv_p_coeff);
    % Integral Compensation
    rhs_coeff = rhs_coeff-settings.Mint'*(settings.Mint*rhs_coeff-settings.int_const/2/pi*(-K2/alpha(k)))/settings.MintSq;
    
    ucoeff = helmholtz_coeff(rhs_coeff, -K2/alpha(k), n, m);
    
    %    Plot the solution every 25 time steps
    %     if ( mod(i, 5) == 0 )
    %         u=spherefun.coeffs2spherefun(transpose(reshape(ucoeff,m,n)));
    %         contour(u,[0:0.025:0.4]);
    %         title(sprintf('Time %2.5f',i*dt)), snapnow
    %     end
    Sf(i)=real(settings.Mint*ucoeff*2*pi);
    
end


%% Surface Integral Conservation check
% figure;
% plot([dt:dt:tfinal],Sf);

%% Translate back to Sphere for Post-Processing
u=spherefun.coeffs2spherefun(transpose(reshape(ucoeff,m,n)));

n_phi=32; % Has to be even for FFT. 2^N recommended
n_theta=101; % Had better to be 1+(multiples of 5,4,3 or 2) for Newton-Cot

dtheta=(pi/(n_theta-1));
dphi=2*pi/(n_phi);

theta=(0:dtheta:pi)';
phi=0:dphi:(2*pi-dphi);


figure;
contour(phi,theta,u(phi,theta));



