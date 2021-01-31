function ucoeff=time_relaxed_Linv(Mvor,Mgyro,Mlap,S,forcing,Mint,MintSq,helm,ucoeff)
%% Initialisation
N_mesh=numel(S);

if nargin == 8
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
Mgyro=gpuArray(Mgyro);
Mlap=gpuArray(Mlap);
S=gpuArray(S);
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
epsilon=1e-7;
N_check=25;
for ii=1:400
for i=1:(N_check-1)
    %k1
    adv_coeff=S.*(Mvor*ucoeff)+Mgyro*ucoeff;
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
    adv_coeff=S.*(Mvor*ucoeff)+Mgyro*ucoeff;
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
    
    adv_coeff=S.*(Mvor*ucoeff)+Mgyro*ucoeff;
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
    adv_coeff=S.*(Mvor*ucoeff)+Mgyro*ucoeff;
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
    adv_coeff=S.*(Mvor*ucoeff)+Mgyro*ucoeff;
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
    
    adv_coeff=S.*(Mvor*ucoeff)+Mgyro*ucoeff;
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
    err=gather(max(sqrt(sum(abs(ucoeff-ucoeffp).^2,2))));
    if err<epsilon || isnan(err) 
        break;
    end
%         errMp1=gather(max(abs(Mint*(Mp1*(ucoeff-ucoeffp)))))*2*pi/helm.dt;
%         errMp3=gather(max(abs(Mint*(Mp3*(ucoeff-ucoeffp)))))*2*pi/helm.dt;
%         if max(errMp1,errMp3)<epsilon || isnan(errMp1) 
%             break;
%         end
        
% disp([num2str(ii*N_check) '    ' num2str(err) '  ' num2str(errMp1) '   ' num2str(errMp3) '   ' num2str(gather(max(abs(Mint*ucoeff)))*2*pi)]);
end

disp([num2str(ii*N_check) '    ' num2str(err) '     ' num2str(gather(max(abs(Mint*ucoeff)))*2*pi)]);
ucoeff=gather(ucoeff);
end