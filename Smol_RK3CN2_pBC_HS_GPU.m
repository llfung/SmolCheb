%% Time Stepping solver for the Full Smoluchowski Equation (RK3-CN2)
% Loosely based on Spherefun, this solve uses Double Fourier Sphere (DFS)
% method to tranform the orientational space and time-marched in the DFS 
% space. Time stepping is semi-implicit, with Advection terms in RK3 and 
% Laplacian term in CN2. Implicit matrix inversion is done using the
% Spherefun Helmholtz Solver (only slightly modified to remove
% chebfun-based activiities). 

%% Setting up
%Saving to settings struct
settings.beta=beta;
settings.B=B;
settings.Vc=Vc;
settings.n=n;
settings.m=m;
settings.diff_const=diff_const;
settings.dt=dt;
settings.d_spatial=dz;
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

% RK3 coeff and constants
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];

% Preparing Constants
K2 = (1/(dt*diff_const));         % Helmholtz frequency for BDF1

%% Initialising Matrices
[settings,Mvor,Mgyro,Mlap,Mlap_inv,Rdz,Rd2z,Mp1,Mp3,~,Mp3sq]=all_mat_gen(settings);

% mats=struct('Mint',settings.Mint,'S_profile',S_profile,'Mvor',Mvor,'Mgyro',Mgyro,'Mlap',Mlap_inv,...
%     'Mp1',Mp1,'Mp3',Mp3,'Rdz',Rdz,'Rd2z',Rd2z); settings_CPU=settings; % If CPU PS is used.

Mint=gpuArray(settings.Mint);
settings.Mint=Mint;
MintSq=gpuArray(settings.MintSq);

Kp=settings.Kp;

% GPU stuff
S_profile=gpuArray(S_profile);
Kp=gpuArray(Kp);

mKp_alpha1=gpuArray(-(Kp/alpha(1)));
mKp_alpha2=gpuArray(-(Kp/alpha(2)));
mKp_alpha3=gpuArray(-(Kp/alpha(3)));
mK2_alpha1=gpuArray(-K2/alpha(1));malpha1_K2=1/mK2_alpha1;
mK2_alpha2=gpuArray(-K2/alpha(2));malpha2_K2=1/mK2_alpha2;
mK2_alpha3=gpuArray(-K2/alpha(3));malpha3_K2=1/mK2_alpha3;
gamma_alpha1=gpuArray(gamma(1)/alpha(1)/diff_const);
gamma_alpha2=gpuArray(gamma(2)/alpha(2)/diff_const);
gamma_alpha3=gpuArray(gamma(3)/alpha(3)/diff_const);
rho_alpha2=gpuArray(rho(2)/alpha(2)/diff_const);
rho_alpha3=gpuArray(rho(3)/alpha(3)/diff_const);

% Advection
% Mvor=gpuArray(complex(adv_vor_mat(settings)));
% Mgyro=gpuArray(complex(settings.beta*adv_gyro_mat(settings)));
%For strange behaviour in MATLAB ver < R2020
Mvor=gpuArray(sparse(complex(full(Mvor))));
Mgyro=gpuArray(sparse(complex(full(Mgyro))));

%Laplacian
Mlap=gpuArray(sparse(complex(Mlap)));
Mlap_inv=gpuArray(sparse(complex(Mlap_inv)));

helm=helmholtz_genGPU( n, m);
helm_inv_k1=helmholtz_precalGPU( -K2/alpha(1),helm);
helm_inv_k2=helmholtz_precalGPU( -K2/alpha(2),helm);
helm_inv_k3=helmholtz_precalGPU( -K2/alpha(3),helm);

%Dz
Rdz=gpuArray(Rdz);
Rd2z=gpuArray(Rd2z);

%p1
Mp1 = gpuArray(sparse(complex(full(Mp1))));
%p3
Mp3 = gpuArray(sparse(complex(full(Mp3))));
%p3p3
Mp3sq = gpuArray(sparse(complex(full(Mp3sq))));

%Swimming and sedimentation
MSwim=Vc*Mp3-Vsmin*gpuArray(speye(n*m))-Vsvar*Mp3sq;

mats=struct('Mint',settings.Mint,'S_profile',S_profile,'Mvor',Mvor,'Mgyro',Mgyro,'Mlap',Mlap_inv,...
    'Mp1',Mp1,'Mp3',Mp3,'Rdz',Rdz,'Rd2z',Rd2z);  % If GPU PS is used.

%% Initialise Recorded values
cell_den=NaN(floor(nsteps/saving_rate3),N_mesh);

% PS=PS_RunTime('z','inv',mats,settings_CPU,saving_rate1,saving_rate2);
PS=PS_RunTime('z','invGPU_w_fdt',mats,settings,saving_rate1,saving_rate2);

%% Time-Stepping (RK3-CN2)
ucoeff=gpuArray(complex(ucoeff0));
adv_p_coeff     =gpuArray(complex(zeros(n*m,N_mesh)));
adv_comb_coeff  =gpuArray(complex(zeros(n*m,N_mesh)));
ucoeff_previous2=complex(NaN(n*m,N_mesh,3));

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dz;

for i = 1:nsteps
    %% RK step 1
    % k=1;
    % Par-For Version
    dzu_coeff=ucoeff*Rdz;
    dz2u_coeff=ucoeff*Rd2z;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    
    swim_coeff=MSwim*dzu_coeff;
    
    DT_coeff=DT*dz2u_coeff;
    
    adv_p_coeff=adv_coeff+swim_coeff-DT_coeff;
    
    rhs_coeff = mK2_alpha1*ucoeff-lap_coeff+gamma_alpha1*adv_p_coeff...
             +mKp_alpha1*(int_const-Nint_loc)*(Mint'.*ucoeff);

    F=permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,malpha1_K2*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,malpha1_K2*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k1*reshape(F,helm.n*helm.m,N_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);  
   
    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dz;
    
    %% RK step 2
    % k=2;
    dzu_coeff=ucoeff*Rdz;
    dz2u_coeff=ucoeff*Rd2z;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    
    swim_coeff=MSwim*dzu_coeff;
    
    DT_coeff=DT*dz2u_coeff;
    
    adv_comb_coeff=adv_coeff+swim_coeff-DT_coeff;
    
    rhs_coeff = mK2_alpha2*ucoeff-lap_coeff+gamma_alpha2*adv_comb_coeff+rho_alpha2*adv_p_coeff...
            +mKp_alpha2*(int_const-Nint_loc)*(Mint'.*ucoeff);

    F=permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,malpha2_K2*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,malpha2_K2*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k2*reshape(F,helm.n*helm.m,N_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);     

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dz;
    
    %% RK step 3
    % k=3;
    dzu_coeff=ucoeff*Rdz;
    dz2u_coeff=ucoeff*Rd2z;
    adv_p_coeff=adv_comb_coeff;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    
    swim_coeff=MSwim*dzu_coeff;
    
    DT_coeff=DT*dz2u_coeff;
    
    adv_comb_coeff=adv_coeff+swim_coeff-DT_coeff;
    
    rhs_coeff = mK2_alpha3*ucoeff-lap_coeff+gamma_alpha3*adv_comb_coeff+rho_alpha3*adv_p_coeff...
            +mKp_alpha3*(int_const-Nint_loc)*(Mint'.*ucoeff);

    F=permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,malpha3_K2*helm.enG,F(:,helm.k,:));

    F = pagefun(@mtimes,malpha3_K2*helm.L2G, F);

    F(helm.floorm+1,helm.k,:)=int_constj;

    CFS = helm_inv_k3*reshape(F,helm.n*helm.m,N_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);     

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dz;
    
    %% Saving for Post-Processing    
    % Saving full Psi and it time derivative
    PS=PS.RunTimeCall(ucoeff,i);
    
    % Saving Cell Density
    if ( mod(i, saving_rate3) == 0 )
        cell_den(i/saving_rate3,:)=gather(cell_den_loc);
    end    
end