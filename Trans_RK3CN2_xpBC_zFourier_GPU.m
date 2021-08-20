%% Full Smoluchowski Time Stepping solver for the Fokker-Planck Equation (RK3-CN2)
% Loosely based on Spherefun, this solve uses Double Fourier Sphere (DFS)
% method to tranform the orientational space and time-marched in the DFS 
% space. Time stepping is semi-implicit, with Advection terms in RK3 and 
% Laplacian term in CN2. Implicit matrix inversion is done using the
% Spherefun Helmholtz Solver (only slightly modified to remove
% chebfun-based activiities). 

%% Setting up
% RK3 coeff and constants
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];

% Preparing Constants
K2 = (1/(dt*diff_const));         % Helmholtz frequency for BDF1

%% Initialising Matrices
[settings,~,~,~,Rdx,Rd2x,~,~,~,~]=all_mat_gen(settings);

Kp=settings.Kp;

% GPU stuff
S_profile=gpuArray(S_profile);
Kp=gpuArray(Kp);int_const=gpuArray(int_const);

Kp_K2=gpuArray(Kp/K2);
gamma1dt=gpuArray(gamma(1)*dt);
gamma2dt=gpuArray(gamma(2)*dt);
gamma3dt=gpuArray(gamma(3)*dt);
rho2dt=gpuArray(rho(2)*dt);
rho3dt=gpuArray(rho(3)*dt);

%Expansion to 2D
%Dx
Rdx=gpuArray(kron(speye(Nz_mesh),Rdx));
Rd2x=gpuArray(kron(speye(Nz_mesh),Rd2x));

%Dz
rowDz=gpuArray(kron(ifftshift((-(Nz_mesh/2):(Nz_mesh/2)-1)*1i*alphaK),ones(1,Nx_mesh)));
rowDzz=gpuArray(kron(ifftshift((-(Nz_mesh/2):(Nz_mesh/2)-1).^2*alphaK^2*(-1)),ones(1,Nx_mesh)));
Nint_row=gpuArray([ones(1,Nx_mesh)*dx*z_width zeros(1,(Nz_mesh-1)*Nx_mesh)]');

%% Initial Condition
ucoeff0=reshape(norm_distr_T,1,[]);

%% Initialise Recorded values
cell_den=NaN(floor(nsteps/saving_rate3),Nx_mesh*Nz_mesh);

%% Time-Stepping (RK3-CN2)
ucoeff=gpuArray(complex(ucoeff0));
adv_p_coeff   =gpuArray(complex(zeros(n*m,Nx_mesh*Nz_mesh)));
adv_comb_coeff=gpuArray(complex(zeros(n*m,Nx_mesh*Nz_mesh)));
ucoeff_previous=gpuArray(complex(NaN(n*m,Nx_mesh*Nz_mesh,3)));
% ucoeff_previous2=gpuArray(complex(NaN(n*m,N_mesh,3)));

    Nint_loc=ucoeff*Nint_row;

for i = 1:nsteps
    %% RK step 1
    %k=1;
    dxu_coeff=ucoeff*Rdx;
    dzu_coeff=ucoeff.*rowDz;

    uadv_coeff=U_profile.*dxu_coeff+W_profile.*dzu_coeff;
    DT_coeff=DTxx*ucoeff*Rd2x+DTzz*ucoeff.*rowDzz+2*DTxz*(ucoeff*Rdx).*rowDz;
    adv_p_coeff=uadv_coeff-DT_coeff;
    ucoeff = ucoeff-gamma1dt*adv_p_coeff...
             +Kp_K2*(int_const-Nint_loc)*ucoeff;
         
    Nint_loc=ucoeff*Nint_row;
    
    %% RK step 2
    %k=2;
    dxu_coeff=ucoeff*Rdx;
    dzu_coeff=ucoeff.*rowDz;
    
    uadv_coeff=U_profile.*dxu_coeff+W_profile.*dzu_coeff;
    DT_coeff=DTxx*ucoeff*Rd2x+DTzz*ucoeff.*rowDzz+2*DTxz*(ucoeff*Rdx).*rowDz;
    adv_comb_coeff=uadv_coeff-DT_coeff;
    ucoeff = ucoeff-gamma2dt*adv_comb_coeff-rho2dt*adv_p_coeff...
            +Kp_K2*(int_const-Nint_loc)*ucoeff;

    Nint_loc=ucoeff*Nint_row;
    
    %% RK step 3
    %k=3;
    adv_p_coeff=adv_comb_coeff;
    dxu_coeff=ucoeff*Rdx;
    dzu_coeff=ucoeff.*rowDz;
    
    uadv_coeff=U_profile.*dxu_coeff+W_profile.*dzu_coeff;
    DT_coeff=DTxx*ucoeff*Rd2x+DTzz*ucoeff.*rowDzz+2*DTxz*(ucoeff*Rdx).*rowDz;
    adv_comb_coeff=uadv_coeff-DT_coeff;
    ucoeff = ucoeff-gamma3dt*adv_comb_coeff-rho3dt*adv_p_coeff...
            +Kp_K2*(int_const-Nint_loc)*ucoeff;
         
    Nint_loc=ucoeff*Nint_row;
    
    %% Saving for Post-Processing    
    % Saving full Psi and it time derivative
%     PS=PS.RunTimeCall(ucoeff,i);
    
    % Saving Cell Density
     if ( mod(i, saving_rate3) == 0 )
        cell_den(i/saving_rate3,:)=gather(ucoeff);
        disp([num2str(i) '/' num2str(nsteps)]);
    end 
end
