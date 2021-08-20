addpath(genpath('core'))
addpath(genpath('core/x_FD'))
addpath(genpath('core/p_DFS'))
addpath(genpath('core/PS_RT'))
gpuDevice(2);

%% Load
load('E:\db\Smol\bearon2011\smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_0.5Pef_dx_448dz_448_m8_n8_dt0.001_tf35.002_PS.mat')
clearvars ng;
%load('smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_1Pef_dx_384dz_384_m8_n8_dt0.00025_tf10_PS.mat');

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
%     +spdiags(transpose(Vc*ex_avg-Vc^2*Vix-Vc*Vuxf),0,N_mesh,N_mesh)*Rdx...
%     +spdiags(transpose(Vc*ez_avg-Vc^2*Viz-Vc*Vuzf),0,N_mesh,N_mesh)*Rdz...
%     +Vc^2*(-Rd2x*spdiags(transpose(Dxx),0,N_mesh,N_mesh)-Rd2z*spdiags(transpose(Dzz),0,N_mesh,N_mesh)...
%     -Rdz*Rdx*spdiags(transpose(Dxz+Dzx),0,N_mesh,N_mesh)...
%     -Rdx*spdiags(transpose(Dxx*Rdx+Dzx*Rdz),0,N_mesh,N_mesh)...
%     -Rdz*spdiags(transpose(Dxz*Rdx+Dzz*Rdz),0,N_mesh,N_mesh));


%save('temp.mat','-v7.3');
Trans_RK3CN2_xpBC_zpBC_GPU;
% ngs=[zeros(1,N_mesh) 1/dx/dz]/[OP ones(N_mesh,1)];

clearvars Msin Mp1 Mp3 Rd Rd2 Rdx Rd2x Rdz Rd2z Mvor Mstrain Mgyro Mlap fac arr;
save('smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_0.5Pef_dx_448dz_448_m8_n8_dt0.001_tf35.002_PS_Allg_Vuf.mat','-v7.3');
%exit

