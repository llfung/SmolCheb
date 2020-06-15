%% Full Smoluchowski Time Stepping solver for the Fokker-Planck Equation (RK3-CN2)
% Loosely based on Spherefun, this solve uses Double Fourier Sphere (DFS)
% method to tranform the orientational space and time-marched in the DFS 
% space. Time stepping is semi-implicit, with Advection terms in RK3 and 
% Laplacian term in CN2. Implicit matrix inversion is done using the
% Spherefun Helmholtz Solver (only slightly modified to remove
% chebfun-based activiities). 

% pc=parcluster('local');
% pc.JobStorageLocation = strcat(getenv('TMPDIR'),'/para_tmp');
% par=parpool(pc,32);

% parpool(20);
clear all;
%% Setting up
% Parameters
dt = 0.02;                  % Time step
tfinal = 10;               % Stopping time
nsteps = ceil(tfinal/dt);   % Number of time steps
m = 16;                     % Spatial discretization - phi (even)
n = 20;                     % Spaptial discretization - theta (even)
N_mesh=101;                 % Spaptial discretization - y
diff_const = 1;             % Diffusion constant
beta=2.2;                   % Gyrotactic time scale
% S=2.5;                      % Shear time scale
Vc=1/5;                       % Swimming Speed (scaled by channel width and Dr) (Pe_s)
Pef=Vc*2;
omg=[0,-1,0];                % Vorticity direction (1,2,3) 

% Run saving settings
saving_rate1=10;
saving_rate2=50;

x_sav_location=[1 11 21 26 31 41 51];

%Saving to settings struct
% settings.S=S;
settings.beta=beta;
settings.n=n;
settings.m=m;
settings.omg1=omg(1);
settings.omg2=omg(2);
settings.omg3=omg(3);


%% x-domain Meshing
dx=2/(N_mesh-1);
x=-1:dx:1;

%% Initial Condition (not recorded)
settings.int_const=1.;
ucoeff0=zeros(n*m,N_mesh);ucoeff0(m*n/2+m/2+1,:)=1/8/pi;

%% Shear Profile
% W_profile=(-cos(pi*x)-1)*Pef;   % W(x)=-cos(pi x)-1
S_profile=pi*sin(pi*x)*Pef; % W(x)=-cos(pi x)-1
S_profile(1)=0;

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
Rdx(:,1)=0;Rdx(:,N_mesh)=0;
Rdx(1:7,     1)=[-49/20;6;-15/2;20/3;-15/4;6/5;-1/6];
Rdx(N_mesh-6:N_mesh,N_mesh)=[1/6;-6/5;15/4;-20/3;15/2;-6;49/20];
Rdx=Rdx/dx;

%p1
Mp1=kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));

%% Initialise Recorded values
cell_den=NaN(floor(nsteps/saving_rate1),N_mesh);

ufull_save=NaN(n*m,N_mesh,floor(nsteps/saving_rate2));

u_xloc_save=NaN(n*m,floor(nsteps/saving_rate1),length(x_sav_location));

%% Time-Stepping (RK3-CN2)
ucoeff=ucoeff0;
adv_p_coeff   =zeros(n*m,N_mesh);
adv_comb_coeff=zeros(n*m,N_mesh);
for i = 1:nsteps
    %% RK step 1
    k=1;
    dxu_coeff=ucoeff*Rdx;
    for j=1:N_mesh
        settings_loc=settings;
        settings_loc.S=S_profile(j);
        adv_coeff=(settings_loc.S/2)*(Mvor*ucoeff(:,j))+Mgyro*ucoeff(:,j);
        adv_coeff=adv_coeff-settings_loc.Mint'*(settings_loc.Mint*adv_coeff)/settings_loc.MintSq;
        
        lap_coeff=Mlap*ucoeff(:,j);
        lap_coeff=lap_coeff-settings_loc.Mint'*(settings_loc.Mint*lap_coeff)/settings_loc.MintSq;
        
        swim_coeff=Vc*Mp1*dxu_coeff(:,j);
        
        adv_p_coeff(:,j)=adv_coeff+swim_coeff;
        if j==1 || j==N_mesh
            adv_col=transpose(reshape(adv_p_coeff(:,j),m,n));
            floorm=floor(m/2);
            if mod(m,2)
            for l=1:floorm
                adv_col(:,floorm+1-l)=(adv_col(:,floorm+1-l)+(-1)^l*adv_col(:,floorm+1+l))/2;
                adv_col(:,floorm+1+l)=(-1)^l*adv_col(:,floorm+1-l);
            end
            else
                adv_col(:,1)=0;
            for l=1:floorm-1
                adv_col(:,floorm+1-l)=(adv_col(:,floorm+1-l)+(-1)^l*adv_col(:,floorm+1+l))/2;
                adv_col(:,floorm+1+l)=(-1)^l*adv_col(:,floorm+1-l);
            end       
            end
            adv_p_coeff(:,j)=reshape(transpose(adv_col),n*m,1);
        end
        
        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_p_coeff(:,j)));
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
    end
    %% RK step 2
    k=2;
    dxu_coeff=ucoeff*Rdx;
    parfor j=1:N_mesh
        settings_loc=settings;
        settings_loc.S=S_profile(j);
%         adv_p_coeff=adv_coeff+swim_coeff;
        adv_coeff=(settings_loc.S/2)*(Mvor*ucoeff(:,j))+Mgyro*ucoeff(:,j);
        adv_coeff=adv_coeff-settings_loc.Mint'*(settings_loc.Mint*adv_coeff)/settings_loc.MintSq;
        
        lap_coeff=Mlap*ucoeff(:,j);
        lap_coeff=lap_coeff-settings_loc.Mint'*(settings_loc.Mint*lap_coeff)/settings_loc.MintSq;
        
        swim_coeff=Vc*Mp1*dxu_coeff(:,j);
        adv_comb_coeff(:,j)=adv_coeff+swim_coeff;
        if j==1 || j==N_mesh
            adv_col=transpose(reshape(adv_comb_coeff(:,j),m,n));
            floorm=floor(m/2);
            if mod(m,2)
            for l=1:floorm
                adv_col(:,floorm+1-l)=(adv_col(:,floorm+1-l)+(-1)^l*adv_col(:,floorm+1+l))/2;
                adv_col(:,floorm+1+l)=(-1)^l*adv_col(:,floorm+1-l);
            end
            else
                adv_col(:,1)=0;
            for l=1:floorm-1
                adv_col(:,floorm+1-l)=(adv_col(:,floorm+1-l)+(-1)^l*adv_col(:,floorm+1+l))/2;
                adv_col(:,floorm+1+l)=(-1)^l*adv_col(:,floorm+1-l);
            end       
            end
            adv_comb_coeff(:,j)=reshape(transpose(adv_col),n*m,1);
        end
        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_comb_coeff(:,j))+rho(k)*adv_p_coeff(:,j)); %#ok<*PFBNS>
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
    end
    %% RK step 3
    k=3;
    dxu_coeff=ucoeff*Rdx;
    adv_p_coeff=adv_comb_coeff;
    parfor j=1:N_mesh
        settings_loc=settings;
        settings_loc.S=S_profile(j);
        
        adv_coeff=(settings_loc.S/2)*(Mvor*ucoeff(:,j))+Mgyro*ucoeff(:,j);
        adv_coeff=adv_coeff-settings_loc.Mint'*(settings_loc.Mint*adv_coeff)/settings_loc.MintSq;
        
        lap_coeff=Mlap*ucoeff(:,j);
        lap_coeff=lap_coeff-settings_loc.Mint'*(settings_loc.Mint*lap_coeff)/settings_loc.MintSq;
        
        swim_coeff=Vc*Mp1*dxu_coeff(:,j);
        adv_comb_coeff(:,j)=adv_coeff+swim_coeff;
        if j==1 || j==N_mesh
            adv_col=transpose(reshape(adv_comb_coeff(:,j),m,n));
            floorm=floor(m/2);
            if mod(m,2)
            for l=1:floorm
                adv_col(:,floorm+1-l)=(adv_col(:,floorm+1-l)+(-1)^l*adv_col(:,floorm+1+l))/2;
                adv_col(:,floorm+1+l)=(-1)^l*adv_col(:,floorm+1-l);
            end
            else
                adv_col(:,1)=0;
            for l=1:floorm-1
                adv_col(:,floorm+1-l)=(adv_col(:,floorm+1-l)+(-1)^l*adv_col(:,floorm+1+l))/2;
                adv_col(:,floorm+1+l)=(-1)^l*adv_col(:,floorm+1-l);
            end       
            end
            adv_comb_coeff(:,j)=reshape(transpose(adv_col),n*m,1);
        end
        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_comb_coeff(:,j))+rho(k)*adv_p_coeff(:,j));
        %     % Integral Compensation
        %     rhs_coeff = rhs_coeff-settings.Mint'*(settings.Mint*rhs_coeff-settings.int_const/2/pi*(-K2/alpha(k)))/settings.MintSq;
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
        
    end
    
    %% Saving for Post-Processing
    if ( mod(i, saving_rate1) == 0 )
        for j=1:length(x_sav_location)
            u_xloc_save(:,i/saving_rate1,j)=ucoeff(:,x_sav_location(j));
        end
        
        cell_den(i/saving_rate1,:)=real(settings.Mint*ucoeff*2*pi);
    end
    
    %    Plot/Save the solution every saving_rate
    if ( mod(i, saving_rate2) == 0 )
        ufull_save(:,:,i/saving_rate2)=ucoeff;
%         u=spherefun.coeffs2spherefun(transpose(reshape(ucoeff,m,n)));
%         plot(u,[0:0.025:0.4]);
%         title(sprintf('Time %2.5f',i*dt)), snapnow
    end
    
end


%% Surface Integral Conservation check
Nint=sum(cell_den,2)*dx;
t1=dt*saving_rate1:dt*saving_rate1:tfinal;
t2=dt*saving_rate2:dt*saving_rate2:tfinal;

% figure;plot(t1,Nint);
% figure;plot(x,Sf);

save('smol_rBC_2-2beta_0-2Vc_0-4Pef_cospi.mat',...
    't1','t2','Nint','cell_den','ufull_save','u_xloc_save','ucoeff','ucoeff0',...
    'settings','x_sav_location','x','dx','dt','diff_const','beta','tfinal',...
    'nsteps','S_profile','N_mesh','n','m','Vc','Pef','omg',...
    'saving_rate1','saving_rate2');
% exit


%% Translate back to Sphere for Post-Processing
% u=spherefun.coeffs2spherefun(transpose(reshape(ucoeff(:,75),m,n)));
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



