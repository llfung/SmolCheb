addpath(genpath('core'))
addpath(genpath('core/x_FD'))
addpath(genpath('core/p_DFS'))
addpath(genpath('core/PS_RT'))

% load('D:\db\Smol\bearon2011\smol_pBC_bearon2011_0.21beta_0B_0.25Vc_0DT_0.5Pef_dx_256dz_256_m8_n8_dt0.01_tf25.02.mat')
load('smol_pBC_bearon2011_0.21beta_0.31B_0.25Vc_0DT_1Pef_dx_320dz_320_m8_n8_dt0.002_tf50.mat')
%% Setting Matrices
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
Mint=kron(fac,[zeros(1,m/2) 1 zeros(1,m/2-1)]);
MintSq=Mint*Mint';
settings.Mint=Mint;
settings.MintSq=MintSq;

settings.Kp=settings.Kp/MintSq/settings.diff_const/settings.dt;

% Advection
Mvor=adv_vor_mat(settings);
Mstrain=settings.B*adv_strain_mat(settings);
Mgyro=settings.beta*adv_gyro_mat(settings);

%Laplacian
Mlap=lap_mat(settings);

Rd=spdiags(ones(Nx_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],Nx_mesh,Nx_mesh);
Rd=spdiags(ones(Nx_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-Nx_mesh+3:-1:-Nx_mesh+1 Nx_mesh-1:-1:Nx_mesh-3],Rd);
Rd=Rd/dx;
Rd2=spdiags(ones(Nx_mesh,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90],[3:-1:-3],Nx_mesh,Nx_mesh);
Rd2=spdiags(ones(Nx_mesh,1)*[1/90 -3/20 3/2 3/2 -3/20 1/90],[-Nx_mesh+3:-1:-Nx_mesh+1 Nx_mesh-1:-1:Nx_mesh-3],Rd2);
Rd2=Rd2/dx/dx;
Rdx=(kron(speye(Nz_mesh),Rd));
Rd2x=(kron(speye(Nz_mesh),Rd2));

Rd=spdiags(ones(Nz_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],Nz_mesh,Nz_mesh);
Rd=spdiags(ones(Nz_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-Nz_mesh+3:-1:-Nz_mesh+1 Nz_mesh-1:-1:Nz_mesh-3],Rd);
Rd=Rd/dz;
Rd2=spdiags(ones(Nz_mesh,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90],[3:-1:-3],Nz_mesh,Nz_mesh);
Rd2=spdiags(ones(Nz_mesh,1)*[1/90 -3/20 3/2 3/2 -3/20 1/90],[-Nz_mesh+3:-1:-Nz_mesh+1 Nz_mesh-1:-1:Nz_mesh-3],Rd2);
Rd2=Rd2/dz/dz;
Rdz=(kron(Rd,speye(Nz_mesh)));
Rd2z=(kron(Rd2,speye(Nz_mesh)));

%p1
Mp1 = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));
%p3
Mp3 = kron(spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n),speye(m));

Msin=(kron(spdiags(0.5i*ones(n,1)*[-1,1], [-1 1], n, n),speye(m)));

%%
pspace_dt=0.015;
helm=helmholtz_genGPU( n, m);
helm.dt=pspace_dt;

%% Getting g(p)
N_mesh=Nx_mesh*Nz_mesh;
nm=n*m;
% g=NaN(nm,N_mesh);

g=time_relaxed_Linv_fd2d(Mvor,Mstrain,Mgyro,Mlap,curl_profile,E_profile,zeros(n*m,1),Mint,MintSq,helm);

%% Calculating drifts and dispersions
ex_avg=real(Mint*Mp1*g*(2*pi));
ez_avg=real(Mint*Mp3*g*(2*pi));

bx_RHS=Mp1*g-ex_avg.*g;
bz_RHS=Mp3*g-ez_avg.*g;
        
inhomo_RHS=Mp1*(g*Rdx)+Mp3*(g*Rdz)-(ex_avg*Rdx+ez_avg*Rdz).*g;

fu_RHS=U_profile.*(g*Rdx)+W_profile.*(g*Rdz);

bx=time_relaxed_Linv_fd2d(Mvor,Mstrain,Mgyro,Mlap,curl_profile,E_profile,bx_RHS,Mint,MintSq,helm);
bz=time_relaxed_Linv_fd2d(Mvor,Mstrain,Mgyro,Mlap,curl_profile,E_profile,bz_RHS,Mint,MintSq,helm);
f_inhomo=time_relaxed_Linv_fd2d(Mvor,Mstrain,Mgyro,Mlap,curl_profile,E_profile,inhomo_RHS,Mint,MintSq,helm);
fu=time_relaxed_Linv_fd2d(Mvor,Mstrain,Mgyro,Mlap,curl_profile,E_profile,fu_RHS,Mint,MintSq,helm);

clearvars Le LHS zero_row bx_RHS bz_RHS inhomo_RHS fu_RHS adv_coeff;
  
        Dxx=real(Mint*(Mp1*reshape(bx,nm,N_mesh))*(2*pi));
        Dxz=real(Mint*(Mp1*reshape(bz,nm,N_mesh))*(2*pi));
        Dzx=real(Mint*(Mp3*reshape(bx,nm,N_mesh))*(2*pi));
        Dzz=real(Mint*(Mp3*reshape(bz,nm,N_mesh))*(2*pi));
        Vix=real(Mint*(Mp1*reshape(f_inhomo,nm,N_mesh))*(2*pi));
        Viz=real(Mint*(Mp3*reshape(f_inhomo,nm,N_mesh))*(2*pi));
        
        Vux=real(Mint*(Mp1*reshape(fu,nm,N_mesh))*(2*pi));
        Vuz=real(Mint*(Mp3*reshape(fu,nm,N_mesh))*(2*pi));

clearvars bx bz f_inhomo fu;
%% Calculating drifts and dispersions
f=ucoeff./real(Mint*ucoeff)/2/pi;
ex=real(Mint*Mp1*ucoeff)./real(Mint*ucoeff);
ez=real(Mint*Mp3*ucoeff)./real(Mint*ucoeff);
zero_row=zeros(1,Nx_mesh*Nz_mesh);k=n*m/2+m/2+1;

bx_RHS=Mp1*f-ex.*f;
bz_RHS=Mp3*f-ez.*f;
        
inhomo_RHS=Mp1*(f*Rdx)+Mp3*(f*Rdz)-(ex*Rdx+ez*Rdz).*f;

fu_RHS=U_profile.*(f*Rdx)+W_profile.*(f*Rdz);

bx=time_relaxed_Linv_fd2d(Mvor,Mstrain,Mgyro,Mlap,curl_profile,E_profile,bx_RHS,Mint,MintSq,helm);
bz=time_relaxed_Linv_fd2d(Mvor,Mstrain,Mgyro,Mlap,curl_profile,E_profile,bz_RHS,Mint,MintSq,helm);
f_inhomo=time_relaxed_Linv_fd2d(Mvor,Mstrain,Mgyro,Mlap,curl_profile,E_profile,inhomo_RHS,Mint,MintSq,helm);
fu=time_relaxed_Linv_fd2d(Mvor,Mstrain,Mgyro,Mlap,curl_profile,E_profile,fu_RHS,Mint,MintSq,helm);

clearvars Le LHS zero_row bx_RHS bz_RHS inhomo_RHS fu_RHS adv_coeff;
  
        Dxxf=real(Mint*(Mp1*reshape(bx,nm,N_mesh))*(2*pi));
        Dxzf=real(Mint*(Mp1*reshape(bz,nm,N_mesh))*(2*pi));
        Dzxf=real(Mint*(Mp3*reshape(bx,nm,N_mesh))*(2*pi));
        Dzzf=real(Mint*(Mp3*reshape(bz,nm,N_mesh))*(2*pi));
        Vixf=real(Mint*(Mp1*reshape(f_inhomo,nm,N_mesh))*(2*pi));
        Vizf=real(Mint*(Mp3*reshape(f_inhomo,nm,N_mesh))*(2*pi));
        
        Vuxf=real(Mint*(Mp1*reshape(fu,nm,N_mesh))*(2*pi));
        Vuzf=real(Mint*(Mp3*reshape(fu,nm,N_mesh))*(2*pi));

clearvars bx bz f_inhomo fu;
%% Steady n_{g,s}
OP=Rdx*spdiags(reshape(U_profile,[],1),0,N_mesh,N_mesh)+Rdz*spdiags(reshape(W_profile,[],1),0,N_mesh,N_mesh)...
    +spdiags(transpose(Vc*ex_avg-Vc^2*Vix-Vc*Vux),0,N_mesh,N_mesh)*Rdx...
    +spdiags(transpose(Vc*ez_avg-Vc^2*Viz-Vc*Vuz),0,N_mesh,N_mesh)*Rdz...
    +Vc^2*(-Rd2x*spdiags(transpose(Dxx),0,N_mesh,N_mesh)-Rd2z*spdiags(transpose(Dzz),0,N_mesh,N_mesh)...
    -Rdz*Rdx*spdiags(transpose(Dxz+Dzx),0,N_mesh,N_mesh)...
    -Rdx*spdiags(transpose(Dxx*Rdx+Dzx*Rdz),0,N_mesh,N_mesh)...
    -Rdz*spdiags(transpose(Dxz*Rdx+Dzz*Rdz),0,N_mesh,N_mesh));

% test
% OP=Rdx*spdiags(reshape(U_profile,[],1),0,N_mesh,N_mesh)+Rdz*spdiags(reshape(W_profile,[],1),0,N_mesh,N_mesh)...
%     +spdiags(transpose(Vc*ex_avg-Vc^2*Vixf),0,N_mesh,N_mesh)*Rdx...
%     +spdiags(transpose(Vc*ez_avg-Vc^2*Vizf),0,N_mesh,N_mesh)*Rdz...
%     +Vc^2*(-Rd2x*spdiags(transpose(Dxxf),0,N_mesh,N_mesh)-Rd2z*spdiags(transpose(Dzzf),0,N_mesh,N_mesh)...
%     -Rdz*Rdx*spdiags(transpose(Dxzf+Dzxf),0,N_mesh,N_mesh)...
%     -Rdx*spdiags(transpose(Dxxf*Rdx+Dzxf*Rdz),0,N_mesh,N_mesh)...
%     -Rdz*spdiags(transpose(Dxzf*Rdx+Dzzf*Rdz),0,N_mesh,N_mesh));

%save('temp.mat','-v7.3');
Trans_RK3CN2_xpBC_zpBC_GPU;
% ngs=[zeros(1,N_mesh) 1/dx/dz]/[OP ones(N_mesh,1)];

clearvars Msin Mp1 Mp3 Rd Rd2 Rdx Rd2x Rdz Rd2z Mvor Mstrain Mgyro Mlap fac arr;
% save('smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_0.5Pef_dx_448dz_448_m8_n8_dt0.001_tf35.002_PS.mat');
save('temp.mat','-v7.3');
% save('D:\db\Smol\bearon2011\smol_pBC_bearon2011_0.21beta_0.31B_0.25Vc_0DT_0.5Pef_dx_384dz_384_m8_n16_dt0.002_tf75.004_PSmod.mat','-v7.3');

%%
function ucoeff=time_relaxed_Linv_fd2d(Mvor,Mstrain,Mgyro,Mlap,curl,E,forcing,Mint,MintSq,helm,ucoeff)
%% Initialisation
N_mesh=numel(curl);

if nargin == 10
    ucoeff=zeros(size(forcing));
    if any(forcing,'all')
        forcing=gpuArray(forcing-Mint'*(Mint*forcing)/MintSq);
        init_const=0;
    else
        init_const=1/2/pi;
        ucoeff(helm.n*helm.m/2+helm.n/2+1,:)=1/4/pi; % Note that helm.n=m;helm.n=m;
    end
else
    if any(forcing,'all')
        init_const=0;
    else
        init_const=1/2/pi;
    end
end

%% RK3 coeff and constants
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];
K2=1/helm.dt;

%% GPU everything
Mvor=gpuArray(Mvor);
Mstrain=gpuArray(Mstrain);
Mgyro=gpuArray(Mgyro);
Mlap=gpuArray(Mlap);
curl=gpuArray(curl);
E=gpuArray(E);
Mint=gpuArray(Mint);
MintSq=gpuArray(MintSq);

ucoeff=gpuArray(ucoeff);
enG=gpuArray(helm.en);
L2G=gpuArray(complex(full(helm.L2)));

helm_inv_k1=helmholtz_precalGPU( -K2/alpha(1),helm);
helm_inv_k2=helmholtz_precalGPU( -K2/alpha(2),helm);
helm_inv_k3=helmholtz_precalGPU( -K2/alpha(3),helm);
helm_inv_k1=gpuArray(helm_inv_k1);
helm_inv_k2=gpuArray(helm_inv_k2);
helm_inv_k3=gpuArray(helm_inv_k3);

Kp=gpuArray(0.0/helm.dt)/MintSq;
mKp_alpha1=gpuArray(-(Kp/alpha(1)));
mKp_alpha2=gpuArray(-(Kp/alpha(2)));
mKp_alpha3=gpuArray(-(Kp/alpha(3)));

mK2_alpha1=gpuArray(-K2/alpha(1));malpha1_K2=1/mK2_alpha1;
mK2_alpha2=gpuArray(-K2/alpha(2));malpha2_K2=1/mK2_alpha2;
mK2_alpha3=gpuArray(-K2/alpha(3));malpha3_K2=1/mK2_alpha3;
gamma_alpha1=gpuArray(gamma(1)/alpha(1));
gamma_alpha2=gpuArray(gamma(2)/alpha(2));
gamma_alpha3=gpuArray(gamma(3)/alpha(3));
rho_alpha2=gpuArray(rho(2)/alpha(2));
rho_alpha3=gpuArray(rho(3)/alpha(3));

%% Loop!
if init_const == 0
    epsilon=1e-4;
else
    epsilon=1e-6;
end
N_check=50;
for ii=1:400
for i=1:(N_check-1)
    %k1
    adv_coeff=curl.*(Mvor*ucoeff)+E.*(Mstrain*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;

    adv_p_coeff=adv_coeff-forcing;
    rhs_coeff = mK2_alpha1*ucoeff-lap_coeff+gamma_alpha1*adv_p_coeff...
        +mKp_alpha1*(init_const-(Mint*ucoeff)).*(Mint'.*ucoeff);
    
    F=permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,malpha1_K2*enG,F(:,helm.k,:));
    F = pagefun(@mtimes,malpha1_K2*L2G, F);
    F(helm.floorm+1,helm.k,:)=int_constj;
    CFS = helm_inv_k1*reshape(F,helm.n*helm.m,N_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);  
    %k2
    adv_coeff=curl.*(Mvor*ucoeff)+E.*(Mstrain*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;

    adv_comb_coeff=adv_coeff-forcing;
    rhs_coeff = mK2_alpha2*ucoeff-lap_coeff+gamma_alpha2*adv_comb_coeff+rho_alpha2*adv_p_coeff...
        +mKp_alpha2*(init_const-(Mint*ucoeff)).*(Mint'.*ucoeff);
    
    F=permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,malpha2_K2*enG,F(:,helm.k,:));
    F = pagefun(@mtimes,malpha2_K2*L2G, F);
    F(helm.floorm+1,helm.k,:)=int_constj;
    CFS = helm_inv_k2*reshape(F,helm.n*helm.m,N_mesh);
    
    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);  
    
    %k3
    adv_p_coeff=adv_comb_coeff;
    
    adv_coeff=curl.*(Mvor*ucoeff)+E.*(Mstrain*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;

    adv_comb_coeff=adv_coeff-forcing;
    rhs_coeff = mK2_alpha3*ucoeff-lap_coeff+gamma_alpha3*adv_comb_coeff+rho_alpha3*adv_p_coeff...
        +mKp_alpha3*(init_const-(Mint*ucoeff)).*(Mint'.*ucoeff);

    F=permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,malpha3_K2*enG,F(:,helm.k,:));
    F = pagefun(@mtimes,malpha3_K2*L2G, F);
    F(helm.floorm+1,helm.k,:)=int_constj;
    CFS = helm_inv_k3*reshape(F,helm.n*helm.m,N_mesh);
    
    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);  
end
    ucoeffp=ucoeff;
    %k1
    adv_coeff=curl.*(Mvor*ucoeff)+E.*(Mstrain*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;

    adv_p_coeff=adv_coeff-forcing;
    rhs_coeff = mK2_alpha1*ucoeff-lap_coeff+gamma_alpha1*adv_p_coeff...
        +mKp_alpha1*(init_const-(Mint*ucoeff)).*(Mint'.*ucoeff);
    
    F=permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,malpha1_K2*enG,F(:,helm.k,:));
    F = pagefun(@mtimes,malpha1_K2*L2G, F);
    F(helm.floorm+1,helm.k,:)=int_constj;
    CFS = helm_inv_k1*reshape(F,helm.n*helm.m,N_mesh);

    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);  
    %k2
    adv_coeff=curl.*(Mvor*ucoeff)+E.*(Mstrain*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;

    adv_comb_coeff=adv_coeff-forcing;
    rhs_coeff = mK2_alpha2*ucoeff-lap_coeff+gamma_alpha2*adv_comb_coeff+rho_alpha2*adv_p_coeff...
        +mKp_alpha2*(init_const-(Mint*ucoeff)).*(Mint'.*ucoeff);
    
    F=permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,malpha2_K2*enG,F(:,helm.k,:));
    F = pagefun(@mtimes,malpha2_K2*L2G, F);
    F(helm.floorm+1,helm.k,:)=int_constj;
    CFS = helm_inv_k2*reshape(F,helm.n*helm.m,N_mesh);
    
    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);  
    
    %k3
    adv_p_coeff=adv_comb_coeff;
    
    adv_coeff=curl.*(Mvor*ucoeff)+E.*(Mstrain*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;

    adv_comb_coeff=adv_coeff-forcing;
    rhs_coeff = mK2_alpha3*ucoeff-lap_coeff+gamma_alpha3*adv_comb_coeff+rho_alpha3*adv_p_coeff...
        +mKp_alpha3*(init_const-(Mint*ucoeff)).*(Mint'.*ucoeff);

    F=permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj = pagefun(@mtimes,malpha3_K2*enG,F(:,helm.k,:));
    F = pagefun(@mtimes,malpha3_K2*L2G, F);
    F(helm.floorm+1,helm.k,:)=int_constj;
    CFS = helm_inv_k3*reshape(F,helm.n*helm.m,N_mesh);
    
    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);  
        
    %TODO: better calculation of norm
    err=gather(max(sqrt(sum(abs(ucoeff-ucoeffp).^2,2))))/helm.dt;
    err2=max(abs(curl.*(Mvor*ucoeff)+E.*(Mstrain*ucoeff)+Mgyro*ucoeff-Mlap*ucoeff-forcing),[],'all'); %TODO: this test does not converge!
    if err<epsilon || isnan(err) 
        break;
    end
%         errMp1=gather(max(abs(Mint*(Mp1*(ucoeff-ucoeffp)))))*2*pi/helm.dt
%         errMp3=gather(max(abs(Mint*(Mp3*(ucoeff-ucoeffp)))))*2*pi/helm.dt
%         if max(errMp1,errMp3)<epsilon || isnan(errMp1) 
%             break;
%         end
        
disp([num2str(ii*N_check) '    ' num2str(err) '     ' num2str(err2) '     ' num2str(gather(max(abs(Mint*ucoeff)))*2*pi)]);
end

disp([num2str(ii*N_check) '    ' num2str(err) '     ' num2str(gather(max(abs(Mint*ucoeff)))*2*pi)]);
ucoeff=gather(ucoeff);
end