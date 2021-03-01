%% Environmental set up
addpath(genpath('core'))
addpath(genpath('core/x_FD'))
addpath(genpath('core/p_DFS'))
addpath(genpath('core/PS_RT'))

%% Setting up
% Parameters
beta=2.2;                   % Gyrotactic time scale
B=0;

m = 16;                     % Spatial discretization - phi (even)
n = 32;                     % Spaptial discretization - theta (even)

omg=[0,-1,0];               % Vorticity direction (1,2,3) 

%Saving to settings struct
settings.beta=beta;
settings.B=B;
settings.n=n;
settings.m=m;
settings.omg1=omg(1);
settings.omg2=omg(2);
settings.omg3=omg(3);
settings.e11=0;
settings.e12=0;
settings.e13=1; % For vertical flow
settings.e22=0;
settings.e23=0;
settings.e33=0;

%% x-domain Meshing
S_profile=-(0:0.05:15); % S_profile = -S(x)= Pef/2*(\partial_z U - \partial_x W)
N_mesh=length(S_profile);
settings.N_mesh=N_mesh;
settings.d_spatial=S_profile(1)-S_profile(2);

%% Initialising Matrices
settings.Kp=0.001;
settings.diff_const=1;
settings.dt=0.01;
[settings,Mvor,Mgyro,Mlap,Rdx,Rd2x,Mp1,Mp3,Mp1p3,~]=all_mat_gen(settings);
Mint=settings.Mint;

%% Finding Small Pe Estimation (CPU)
Msin=kron(spdiags(0.5i*ones(settings.n,1)*[-1,1], [-1 1], settings.n, settings.n),speye(settings.m));
[gall,Linv]=Linv_g(S_profile,Mvor,Mgyro,Mlap,settings.Mint,Msin,n,m);
[exg_smallPe,ezg_smallPe,Dxx_smallPe,Dxz_smallPe,Dzx_smallPe,Dzz_smallPe]=...
Linv_f('x',gall,Linv,Msin,Rdx,Rd2x,Mp1,Mp3,settings,zeros(1,N_mesh),n*m/2+m/2+1);

fs=gall(:,1);
ezfs=real(Mint*Mp3*fs(:,1))*2*pi;
bx_RHS=Mp1*fs*ones(1,N_mesh);
bz_RHS=Mp3*fs*ones(1,N_mesh)-ezfs*fs*ones(1,N_mesh);

zero_row=zeros(1,N_mesh);
k=settings.n*settings.m/2+settings.m/2+1;
bx_RHS=Msin*bx_RHS;bx_RHS(k,:)=zero_row;
temp=sum(bsxfun(@times,Linv,reshape(bx_RHS,1,n*m,N_mesh)),2);
bx=reshape(temp(1:n*m,1,:),n*m,N_mesh);
bz_RHS=Msin*bz_RHS;bz_RHS(k,:)=zero_row;
temp=sum(bsxfun(@times,Linv,reshape(bz_RHS,1,n*m,N_mesh)),2);
bz=reshape(temp(1:n*m,1,:),n*m,N_mesh);

Dxx_analytic=real(Mint*(Mp1*reshape(bx,n*m,N_mesh))*(2*pi));
Dxz_analytic=real(Mint*(Mp1*reshape(bz,n*m,N_mesh))*(2*pi));
Dzx_analytic=real(Mint*(Mp3*reshape(bx,n*m,N_mesh))*(2*pi));
Dzz_analytic=real(Mint*(Mp3*reshape(bz,n*m,N_mesh))*(2*pi));

%% load GTD
name=['./FD/GTD_beta_' num2str(beta) 'B_' num2str(B) '_GTD_lib_Sbase_Comp.mat'];
load(name);

%% Small Pe Cell Density
f=figure('Position',[1038         459         420         300]);
plot(omg(2)*S_profile,Dxx_analytic,'k-',omg(2)*S_profile,Dxx_smallPe,'b-.');
hold on; plot(-S_loop/2,res_array(:,1),'r--'); hold off; % S_loop = Pe_f \partial_x W. -S_loop/2 = S_profile
% hold on; plot(S_profile,Dxx_smallPe,'r:');hold off;
xlabel('S');ylabel('Dxx');xlim([0 15]);
legend('Exact $$D_{c}$$','Small $$Pe_s$$ Asymp. $$D_{g,c}$$','GTD $$D_{GTD}$$','location','northeast','Interpreter','latex');
saveas(f,['Dxx.fig']);
saveas(f,['Dxx.eps'],'epsc');

f=figure('Position',[1038         459         420         300]);
plot(omg(2)*S_profile,Dxz_analytic,'k-',omg(2)*S_profile,Dxz_smallPe,'b-.');
% hold on; plot(-S_loop/2,res_array(:,3),'r:'); hold off;
hold on; plot(-S_loop/2,(res_array(:,3)+res_array(:,7))/2,'r--'); hold off;
xlabel('S');ylabel('Dxz');xlim([0 15]);
legend('Exact $$D_{c}$$','Small $$Pe_s$$ Asymp. $$D_{g,c}$$','GTD $$D_{GTD}$$','location','southeast','Interpreter','latex');
saveas(f,['Dxz.fig']);
saveas(f,['Dxz.eps'],'epsc');

f=figure('Position',[1038         459         420         300]);
plot(omg(2)*S_profile,Dzx_analytic,'k-',omg(2)*S_profile,Dzx_smallPe,'b-.');
% hold on; plot(-S_loop/2,res_array(:,7),'r:'); hold off;
hold on; plot(-S_loop/2,(res_array(:,3)+res_array(:,7))/2,'r--'); hold off;
xlabel('S');ylabel('Dzx');xlim([0 15]);
legend('Exact $$D_{c}$$','Small $$Pe_s$$ Asymp. $$D_{g,c}$$','GTD $$D_{GTD}$$','location','southeast','Interpreter','latex');
saveas(f,['Dzx.fig']);
saveas(f,['Dzx.eps'],'epsc');

f=figure('Position',[1038         459         420         300]);
plot(omg(2)*S_profile,Dzz_analytic,'k-',omg(2)*S_profile,Dzz_smallPe,'b-.');
hold on; plot(-S_loop/2,res_array(:,9),'r--'); hold off;
xlabel('S');ylabel('Dzz');xlim([0 15]);
legend('Exact $$D_{c}$$','Small $$Pe_s$$ Asymp. $$D_{g,c}$$','GTD $$D_{GTD}$$','location','northeast','Interpreter','latex');
saveas(f,['Dzz.fig']);
saveas(f,['Dzz.eps'],'epsc');

