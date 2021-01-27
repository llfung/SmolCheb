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
settings.n=n;
settings.m=m;
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
Mint_CPU=kron(fac,[zeros(1,m/2) 1 zeros(1,m/2-1)]);
Mint=gpuArray(Mint_CPU);
MintSq=Mint*Mint';
settings.Mint=Mint;
settings.MintSq=MintSq;

settings.Kp=Kp/MintSq/diff_const/dt; 
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
% Madv=adv_mat(settings);
% Mvor=gpuArray(complex(adv_vor_mat(settings)));
% Mgyro=gpuArray(complex(settings.beta*adv_gyro_mat(settings)));
%For strange behaviour in MATLAB ver < R2020
Mvor=gpuArray(sparse(complex(full(adv_vor_mat(settings)+B*adv_strain_mat(settings)))));
Mgyro=gpuArray(sparse(complex(full(settings.beta*adv_gyro_mat(settings)))));

%Laplacian
Mlap=gpuArray(sparse(complex(full(lap_mat(settings)))));
helm=helmholtz_genGPU( n, m);
helm_inv_k1=helmholtz_precalGPU( -K2/alpha(1),helm);
helm_inv_k2=helmholtz_precalGPU( -K2/alpha(2),helm);
helm_inv_k3=helmholtz_precalGPU( -K2/alpha(3),helm);

%Dx
Rdx=spdiags(ones(N_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],N_mesh,N_mesh);
Rdx=spdiags(ones(N_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-N_mesh+3:-1:-N_mesh+1 N_mesh-1:-1:N_mesh-3],Rdx);
Rdx=gpuArray(Rdx/dx);
Rd2x=spdiags(ones(N_mesh,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90],[3:-1:-3],N_mesh,N_mesh);
Rd2x=spdiags(ones(N_mesh,1)*[1/90 -3/20 3/2 3/2 -3/20 1/90],[-N_mesh+3:-1:-N_mesh+1 N_mesh-1:-1:N_mesh-3],Rd2x);
Rd2x=gpuArray(Rd2x/dx/dx);

%p1
Mp1 = gpuArray(complex(kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m))));
%p3
Mp3 = gpuArray(sparse(complex(full(kron(spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n),speye(m)))))); %e3
%p1p3
Mp1p3 =  gpuArray(sparse(complex(full(kron(spdiags(.25i*ones(n,1)*[-1,1], [-2 2], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m))))));

%Swimming and sedimentation   
MSwim=Vc*Mp1-Vsvar*Mp1p3;

%% Initialise Recorded values
cell_den=NaN(floor(nsteps/saving_rate3),N_mesh);

%% Time-Stepping (RK3-CN2)
ucoeff=gpuArray(complex(ucoeff0));
adv_p_coeff     =gpuArray(complex(zeros(n*m,N_mesh)));
adv_comb_coeff  =gpuArray(complex(zeros(n*m,N_mesh)));
ucoeff_previous2=complex(NaN(n*m,N_mesh,3));

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dx;

for i = 1:nsteps
    %% RK step 1
    % k=1;
    dxu_coeff=ucoeff*Rdx;
    dx2u_coeff=ucoeff*Rd2x;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    
    swim_coeff=MSwim*dxu_coeff;
    
    DT_coeff=DT*dx2u_coeff;
    
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
    Nint_loc=sum(cell_den_loc,2)*dx;
    
    %% RK step 2
    % k=2;
    dxu_coeff=ucoeff*Rdx;
    dx2u_coeff=ucoeff*Rd2x;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    
    swim_coeff=MSwim*dxu_coeff;
    
    DT_coeff=DT*dx2u_coeff;
    
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
    Nint_loc=sum(cell_den_loc,2)*dx;
    
    %% RK step 3
    % k=3;
    dxu_coeff=ucoeff*Rdx;
    dx2u_coeff=ucoeff*Rd2x;
    adv_p_coeff=adv_comb_coeff;
    
    adv_coeff=S_profile.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
    
    swim_coeff=MSwim*dxu_coeff;
    
    DT_coeff=DT*dx2u_coeff;
    
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
    Nint_loc=sum(cell_den_loc,2)*dx;

    %% Saving for Post-Processing    
    % Saving full Psi and it time derivative
    if ( mod(i, saving_rate2) == 0 )
        ufull_save=gather(ucoeff);
        t=i*dt;
    end
    if ( mod(i, saving_rate2) == 2 )&& i~=2 
        ucoeff_CPU=gather(ucoeff);
        fdt_full_save=((-ucoeff_CPU./(real(Mint_CPU*ucoeff_CPU*2*pi))...
            + ucoeff_previous2(:,:,1)./(real(Mint_CPU*ucoeff_previous2(:,:,1)*2*pi)))/12 ...
            +(ucoeff_previous2(:,:,3)./(real(Mint_CPU*ucoeff_previous2(:,:,3)*2*pi))...
            -ucoeff_previous2(:,:,2)./(real(Mint_CPU*ucoeff_previous2(:,:,2)*2*pi)))*(2/3))/dt;
        udt_full_save=((-ucoeff_CPU...
            + ucoeff_previous2(:,:,1))/12 ...
            +(ucoeff_previous2(:,:,3)...
            -ucoeff_previous2(:,:,2))*(2/3))/dt;
        save(['t' num2str(t) '.mat'],'t','ufull_save','fdt_full_save','udt_full_save');
    end
    if ( mod(i, saving_rate2) == 1 )&& i~=1 
        ucoeff_previous2(:,:,3)=gather(ucoeff);
    end
    if ( mod(i, saving_rate2) == saving_rate2-1 )
        ucoeff_previous2(:,:,2)=gather(ucoeff);
    end
    if ( mod(i, saving_rate2) == saving_rate2-2 )
        ucoeff_previous2(:,:,1)=gather(ucoeff);
    end
    
    % Saving Cell Density
    if ( mod(i, saving_rate3) == 0 )
        cell_den(i/saving_rate3,:)=gather(cell_den_loc);
    end    
end