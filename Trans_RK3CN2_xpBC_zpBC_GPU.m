%% Full Smoluchowski Time Stepping solver for the Fokker-Planck Equation (RK3-CN2)
% Loosely based on Spherefun, this solve uses Double Fourier Sphere (DFS)
% method to tranform the orientational space and time-marched in the DFS 
% space. Time stepping is semi-implicit, with Advection terms in RK3 and 
% Laplacian term in CN2. Implicit matrix inversion is done using the
% Spherefun Helmholtz Solver (only slightly modified to remove
% chebfun-based activiities). 
OP=gpuArray(OP);
%% Setting up
dt_ng = 0.00025;                  % Time step
tfinal_ng = 50;          % Stopping time
nsteps_ng = ceil(tfinal_ng/dt_ng);   % Number of time steps

saving_rate4=4000;
t4=dt_ng*saving_rate4:dt_ng*saving_rate4:tfinal_ng;

% RK3 coeff and constants
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];

% Preparing Constants
K2_ng = (1/(dt_ng*diff_const));         % Helmholtz frequency for BDF1

%% Initialising Matrices

% GPU stuff
U_profile=gpuArray(U_profile);
W_profile=gpuArray(W_profile);
Kp=gpuArray(Kp);int_const=gpuArray(int_const);

Kp_K2=gpuArray(Kp/K2_ng/1000);
gamma1dt=gpuArray(gamma(1)*dt_ng);
gamma2dt=gpuArray(gamma(2)*dt_ng);
gamma3dt=gpuArray(gamma(3)*dt_ng);
rho2dt=gpuArray(rho(2)*dt_ng);
rho3dt=gpuArray(rho(3)*dt_ng);

%% Initial Condition
ng_t0=ones(1,Nx_mesh*Nz_mesh)/4;

%% Initialise Recorded values
ng=NaN(floor(nsteps_ng/saving_rate4),Nx_mesh*Nz_mesh);

%% Time-Stepping (RK3-CN2)
ng_t=gpuArray(ng_t0);
adv_p_coeff   =gpuArray((zeros(n*m,Nx_mesh*Nz_mesh)));
adv_comb_coeff=gpuArray((zeros(n*m,Nx_mesh*Nz_mesh)));
%ucoeff_previous=gpuArray((NaN(n*m,Nx_mesh*Nz_mesh,3)));
% ucoeff_previous2=gpuArray(complex(NaN(n*m,N_mesh,3)));

    Nint_loc=sum(ng_t)*dx*dz;

for i = 1:nsteps_ng
    %% RK step 1
    %k=1

    adv_p_coeff=ng_t*OP;
    ng_t = ng_t-gamma1dt*adv_p_coeff...
             +Kp_K2*(int_const-Nint_loc)*ng_t;
         
    Nint_loc=sum(ng_t)*dx*dz;
    
    %% RK step 2
    %k=2;
    
    adv_comb_coeff=ng_t*OP;
    ng_t = ng_t-gamma2dt*adv_comb_coeff-rho2dt*adv_p_coeff...
            +Kp_K2*(int_const-Nint_loc)*ng_t;

    Nint_loc=sum(ng_t)*dx*dz;
    
    %% RK step 3
    %k=3;
    adv_p_coeff=adv_comb_coeff;

    adv_comb_coeff=ng_t*OP;
    ng_t = ng_t-gamma3dt*adv_comb_coeff-rho3dt*adv_p_coeff...
            +Kp_K2*(int_const-Nint_loc)*ng_t;
         
    Nint_loc=sum(ng_t)*dx*dz;
    
    %% Saving for Post-Processing    
    % Saving full Psi and it time derivative
%     PS=PS.RunTimeCall(ucoeff,i);
    
    % Saving Cell Density
     if ( mod(i, saving_rate4) == 0 )
         temp=gather(ng_t);
         if any(isnan(temp))
             error('NaN in ng');
         else
            ng(i/saving_rate4,:)=temp;
         end
        disp([num2str(i) '/' num2str(nsteps_ng)]);
    end 
end

Kp=gather(Kp);int_const=gather(int_const);
U_profile=gather(U_profile);
W_profile=gather(W_profile);
clearvars OP adv_p_coeff adv_comb_coeff temp i Nint_loc int_const ng_t K2_ng Kp_K2 gamma1dt gamma2dt gamma3dt rho2dt rho3dt alpha gamma rho;
