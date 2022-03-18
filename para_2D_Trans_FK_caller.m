addpath(genpath('core'))
addpath(genpath('core/x_FD'))
addpath(genpath('core/p_DFS'))
addpath(genpath('core/PS_RT'))
gpuDevice(2);

%% Load
load('smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_0.5Pef_dx_448dz_448_m8_n8_dt0.001_tf35.002_PS.mat');

%% Setup
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

Kp=0;int_const=1;
%% Fokker-Planck model
%p1
Mp1 = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));
%p3
Mp3 = kron(spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n),speye(m));
%p1p3
Mp1p3 = kron(spdiags(.25i*ones(n,1)*[-1,1], [-2 2], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));
%p3p3
Mp3sq= kron(spdiags(ones(n,1)*[.25,.5,.25], [-2 0 2], n, n),speye(m));
%p1p1
Mp1sq= Mp1*Mp1;

Dxx_FK= real(Mint*Mp1sq*g*(2*pi))-ex_avg.^2;
Dxz_FK= real(Mint*Mp1p3*g*(2*pi))-ex_avg.*ez_avg;
Dzz_FK= real(Mint*Mp3sq*g*(2*pi))-ez_avg.^2;

fac=mean(Dxx)/mean(Dxx_FK);
Dxx_FK=Dxx_FK*fac;
Dxz_FK=Dxz_FK*fac;
Dzz_FK=Dzz_FK*fac;

Dzx_FK= Dxz_FK;

%% Steady n_{g,s}
OP=Rdx*spdiags(reshape(U_profile,[],1),0,N_mesh,N_mesh)+Rdz*spdiags(reshape(W_profile,[],1),0,N_mesh,N_mesh)...
    +spdiags(transpose(Vc*ex_avg),0,N_mesh,N_mesh)*Rdx...
    +spdiags(transpose(Vc*ez_avg),0,N_mesh,N_mesh)*Rdz...
    +Vc^2*(-Rd2x*spdiags(transpose(Dxx_FK),0,N_mesh,N_mesh)-Rd2z*spdiags(transpose(Dzz_FK),0,N_mesh,N_mesh)...
    -Rdz*Rdx*spdiags(transpose(Dxz_FK+Dzx_FK),0,N_mesh,N_mesh)...
    -Rdx*spdiags(transpose(Dxx_FK*Rdx+Dzx_FK*Rdz),0,N_mesh,N_mesh)...
    -Rdz*spdiags(transpose(Dxz_FK*Rdx+Dzz_FK*Rdz),0,N_mesh,N_mesh));

% test
% OP=Rdx*spdiags(reshape(U_profile,[],1),0,N_mesh,N_mesh)+Rdz*spdiags(reshape(W_profile,[],1),0,N_mesh,N_mesh)...
%     +spdiags(transpose(Vc*ex_avg-Vc^2*Vix-Vc*Vuxf),0,N_mesh,N_mesh)*Rdx...
%     +spdiags(transpose(Vc*ez_avg-Vc^2*Viz-Vc*Vuzf),0,N_mesh,N_mesh)*Rdz...
%     +Vc^2*(-Rd2x*spdiags(transpose(Dxx),0,N_mesh,N_mesh)-Rd2z*spdiags(transpose(Dzz),0,N_mesh,N_mesh)...
%     -Rdz*Rdx*spdiags(transpose(Dxz+Dzx),0,N_mesh,N_mesh)...
%     -Rdx*spdiags(transpose(Dxx*Rdx+Dzx*Rdz),0,N_mesh,N_mesh)...
%     -Rdz*spdiags(transpose(Dxz*Rdx+Dzz*Rdz),0,N_mesh,N_mesh));


%save('temp.mat','-v7.3');
Trans_RK3CN2_xpBC_zpBC_GPU;
% ngs=[zeros(1,N_mesh) 1/dx/dz]/[OP ones(N_mesh,1)];
ng_FK=ng;

%% Saving
load('smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_0.5Pef_dx_448dz_448_m8_n8_dt0.001_tf35.002_PS.mat','ng');
clearvars Msin Mp1 Mp3 Rd Rd2 Rdx Rd2x Rdz Rd2z Mvor Mstrain Mgyro Mlap fac arr;
save('smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_0.5Pef_dx_448dz_448_m8_n8_dt0.001_tf35.002_PS_FK.mat','-v7.3');
% save('smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_0.5Pef_dx_448dz_448_m8_n8_dt0.001_tf35.002_PS_Allg_Vuf.mat','-v7.3');
%exit

