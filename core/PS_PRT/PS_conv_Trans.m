addpath(genpath('core'))
addpath(genpath('core/x_FD'))
addpath(genpath('core/p_DFS'))
addpath(genpath('core/PS_RT'))

load('E:\db\Smol\bearon2011\smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_0.5Pef_dx_448dz_448_m8_n8_dt0.001_tf35.002.mat')
% load('smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_0.5Pef_dx_448dz_448_m8_n8_dt0.001_tf35.002.mat');
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

%% Getting g(p)
N_mesh=Nx_mesh*Nz_mesh;
nm=n*m;
g=NaN(nm,N_mesh);

for j=1:N_mesh
    adv_coeff=curl_profile(j)*(Mvor)+E_profile(j)*(Mstrain)+Mgyro;
    Le=full(Msin*gather(adv_coeff-Mlap));
    Le(n*m/2+m/2+1,:)=Mint;

    g(:,j)=Le\[zeros(n*m/2+m/2,1);1/2/pi;zeros(n*m/2-m/2-1,1)];
end

%% Calculating drifts and dispersions
zero_row=zeros(1,Nx_mesh*Nz_mesh);k=n*m/2+m/2+1;
ex_avg=real(Mint*Mp1*g*(2*pi));
ez_avg=real(Mint*Mp3*g*(2*pi));

bx_RHS=Mp1*g-ex_avg.*g;
bz_RHS=Mp3*g-ez_avg.*g;
        
inhomo_RHS=Mp1*(g*Rdx)+Mp3*(g*Rdz)-(ex_avg*Rdx+ez_avg*Rdz).*g;

fu_RHS=U_profile.*(g*Rdx)+W_profile.*(g*Rdz);

bx_RHS=Msin*bx_RHS;bx_RHS(k,:)=zero_row;
bz_RHS=Msin*bz_RHS;bz_RHS(k,:)=zero_row;
inhomo_RHS=Msin*inhomo_RHS;inhomo_RHS(k,:)=zero_row;
fu_RHS=Msin*fu_RHS;fu_RHS(k,:)=zero_row;

bx=NaN(nm,N_mesh);
bz=NaN(nm,N_mesh);
f_inhomo=NaN(nm,N_mesh);
fu=NaN(nm,N_mesh);

for j=1:N_mesh
    Le=full(Msin*gather(curl_profile(j)*(Mvor)+E_profile(j)*(Mstrain)+Mgyro-Mlap));
    Le(n*m/2+m/2+1,:)=Mint;
    
    LHS=Le\[bx_RHS(:,j) bz_RHS(:,j) inhomo_RHS(:,j) fu_RHS(:,j)];

    bx(:,j)=LHS(:,1);
    bz(:,j)=LHS(:,2);
    f_inhomo(:,j)=LHS(:,3);
    fu(:,j)=LHS(:,4);
end
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

bx_RHS=Msin*bx_RHS;bx_RHS(k,:)=zero_row;
bz_RHS=Msin*bz_RHS;bz_RHS(k,:)=zero_row;
inhomo_RHS=Msin*inhomo_RHS;inhomo_RHS(k,:)=zero_row;
fu_RHS=Msin*fu_RHS;fu_RHS(k,:)=zero_row;

nm=n*m;N_mesh=Nx_mesh*Nz_mesh;
bx=NaN(nm,N_mesh);
bz=NaN(nm,N_mesh);
f_inhomo=NaN(nm,N_mesh);
fu=NaN(nm,N_mesh);

for j=1:N_mesh
    Le=full(Msin*gather(curl_profile(j)*(Mvor)+E_profile(j)*(Mstrain)+Mgyro-Mlap));
    Le(n*m/2+m/2+1,:)=Mint;
    
    LHS=Le\[bx_RHS(:,j) bz_RHS(:,j) inhomo_RHS(:,j) fu_RHS(:,j)];

    bx(:,j)=LHS(:,1);
    bz(:,j)=LHS(:,2);
    f_inhomo(:,j)=LHS(:,3);
    fu(:,j)=LHS(:,4);
end
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
save('smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_0.5Pef_dx_448dz_448_m8_n8_dt0.001_tf35.002_PS_noVu.mat','-v7.3');
