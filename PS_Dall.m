gpuDevice(2);
%% Setting up
% Parameters
beta=2.2;                   % Gyrotactic time scale
B=0;

m = 16;                     % Spatial discretization - phi (even)
n = 32;                     % Spaptial discretization - theta (even)

omg=[0,1,0];               % Vorticity direction (1,2,3) 

%Saving to settings struct
settings.beta=beta;
settings.n=n;
settings.m=m;
settings.omg1=omg(1);
settings.omg2=omg(2);
settings.omg3=omg(3);
settings.e11=0;
settings.e12=0;
settings.e13=-1; % For vertical flow
settings.e22=0;
settings.e23=0;
settings.e33=0;

%% x-domain Meshing
S_profile=0:0.05:15; % S_profile = S(x)= Pef/2*(\partial_z U - \partial_x W)
N_mesh=length(S_profile);

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
% Mint=gpuArray(kron(fac,[zeros(1,m/2) 1 zeros(1,m/2-1)]));
Mint=kron(fac,[zeros(1,m/2) 1 zeros(1,m/2-1)]);
MintSq=Mint*Mint';
settings.Mint=Mint;
settings.MintSq=MintSq;

Mvor=adv_vor_mat(settings)+B*adv_strain_mat(settings);
Mgyro=settings.beta*adv_gyro_mat(settings);

%Laplacian
Mlap=lap_mat(settings);
 
%p1
Mp1 = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));

%p3
Mp3 = kron(spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n),speye(m)); %e3

%% Finding Small Pe Estimation (CPU)
exg_smallPe=NaN(1,N_mesh);
ezg_smallPe=NaN(1,N_mesh);
Dxx_smallPe=NaN(1,N_mesh);
Dzx_smallPe=NaN(1,N_mesh);
Dxz_smallPe=NaN(1,N_mesh);
Dzz_smallPe=NaN(1,N_mesh);
Dxx_analytic=NaN(1,N_mesh);
Dzx_analytic=NaN(1,N_mesh);
Dxz_analytic=NaN(1,N_mesh);
Dzz_analytic=NaN(1,N_mesh);

gall=NaN(n*m,N_mesh);

fs=time_relaxed_Linv(Mvor,Mgyro,Mlap,0,Mint,MintSq,n,m,0.01);
ezfs=real(Mint*Mp3*fs(1:n*m,1))*2*pi;

for j=1:N_mesh
dt=0.075/abs(S_profile(j)+7.5);
g=time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile(j),Mint,MintSq,n,m,dt);

exg_smallPe(j)=real(Mint*Mp1*g(1:n*m,1))*2*pi;
ezg_smallPe(j)=real(Mint*Mp3*g(1:n*m,1))*2*pi;

% Le=S_profile(j)*Mvor+Mgyro-Mlap;
bx_RHS=Mp1*g-exg_smallPe(j)*g;
bx=time_relaxed_Linv_f(Mvor,Mgyro,Mlap,S_profile(j),bx_RHS,Mint,MintSq,n,m,dt);

bz_RHS=Mp3*g-ezg_smallPe(j)*g;
bz=time_relaxed_Linv_f(Mvor,Mgyro,Mlap,S_profile(j),bz_RHS,Mint,MintSq,n,m,dt);

bx_RHS=Mp1*fs;
bx_analytic=time_relaxed_Linv_f(Mvor,Mgyro,Mlap,S_profile(j),bx_RHS,Mint,MintSq,n,m,dt);

bz_RHS=Mp3*fs-ezfs*fs;
bz_analytic=time_relaxed_Linv_f(Mvor,Mgyro,Mlap,S_profile(j),bz_RHS,Mint,MintSq,n,m,dt);

Dxx_smallPe(j)=real(Mint*Mp1*bx(1:n*m,1))*2*pi;
Dzx_smallPe(j)=real(Mint*Mp3*bx(1:n*m,1))*2*pi;
Dxz_smallPe(j)=real(Mint*Mp1*bz(1:n*m,1))*2*pi;
Dzz_smallPe(j)=real(Mint*Mp3*bz(1:n*m,1))*2*pi;
% 
Dxx_analytic(j)=real(Mint*Mp1*bx_analytic(1:n*m,1))*2*pi;
Dzx_analytic(j)=real(Mint*Mp3*bx_analytic(1:n*m,1))*2*pi;
Dxz_analytic(j)=real(Mint*Mp1*bz_analytic(1:n*m,1))*2*pi;
Dzz_analytic(j)=real(Mint*Mp3*bz_analytic(1:n*m,1))*2*pi;

gall(:,j)=g;
end
%% load GTD
name=['./FD/GTD_beta_' num2str(beta) 'B_' num2str(B) '_GTD_lib_Sbase_Comp.mat'];
load(name);

%% Small Pe Cell Density
f=figure('Position',[1038         459         420         300]);
plot(S_profile,Dxx_analytic,'k-',S_profile,Dxx_smallPe,'b-.');
hold on; plot(-S_loop/2,res_array(:,1),'r--'); hold off; % S_loop = Pe_f \partial_x W. -S_loop/2 = S_profile
% hold on; plot(S_profile,Dxx_smallPe,'r:');hold off;
xlabel('S');ylabel('Dxx');xlim([0 15]);
legend('Exact $$D_{c}$$','Small $$Pe_s$$ Asymp. $$D_{g,c}$$','GTD $$D_{GTD}$$','location','northeast','Interpreter','latex');
saveas(f,['Dxx.fig']);
saveas(f,['Dxx.eps'],'epsc');

f=figure('Position',[1038         459         420         300]);
plot(S_profile,Dxz_analytic,'k-',S_profile,Dxz_smallPe,'b-.');
% hold on; plot(-S_loop/2,res_array(:,3),'r:'); hold off;
hold on; plot(-S_loop/2,(res_array(:,3)+res_array(:,7))/2,'r--'); hold off;
xlabel('S');ylabel('Dxz');xlim([0 15]);
legend('Exact $$D_{c}$$','Small $$Pe_s$$ Asymp. $$D_{g,c}$$','GTD $$D_{GTD}$$','location','southeast','Interpreter','latex');
saveas(f,['Dxz.fig']);
saveas(f,['Dxz.eps'],'epsc');

f=figure('Position',[1038         459         420         300]);
plot(S_profile,Dzx_analytic,'k-',S_profile,Dzx_smallPe,'b-.');
% hold on; plot(-S_loop/2,res_array(:,7),'r:'); hold off;
hold on; plot(-S_loop/2,(res_array(:,3)+res_array(:,7))/2,'r--'); hold off;
xlabel('S');ylabel('Dzx');xlim([0 15]);
legend('Exact $$D_{c}$$','Small $$Pe_s$$ Asymp. $$D_{g,c}$$','GTD $$D_{GTD}$$','location','southeast','Interpreter','latex');
saveas(f,['Dzx.fig']);
saveas(f,['Dzx.eps'],'epsc');

f=figure('Position',[1038         459         420         300]);
plot(S_profile,Dzz_analytic,'k-',S_profile,Dzz_smallPe,'b-.');
hold on; plot(-S_loop/2,res_array(:,9),'r--'); hold off;
xlabel('S');ylabel('Dzz');xlim([0 15]);
legend('Exact $$D_{c}$$','Small $$Pe_s$$ Asymp. $$D_{g,c}$$','GTD $$D_{GTD}$$','location','northeast','Interpreter','latex');
saveas(f,['Dzz.fig']);
saveas(f,['Dzz.eps'],'epsc');

%%
function g=time_relaxed_Linv(Mvor,Mgyro,Mlap,S,Mint,MintSq,n,m,dt)
persistent ucoeff
%% Initialise Recorded values
if isempty(ucoeff)
    ucoeff=zeros(n*m,1);
    ucoeff(m*n/2+m/2+1,:)=1/4/pi;
else
    ucoeff=ucoeff-Mint'*(Mint*ucoeff-1/2/pi)/MintSq;
end
% 
%     ucoeff=zeros(n*m,1);
%     ucoeff(m*n/2+m/2+1,:)=1/4/pi;
    
Madv=S*(Mvor)+Mgyro;

%% Loop!
ucoeffp=zeros(n*m,1);
% adv_coeff=Madv*ucoeff;
% lap_coeff=Mlap*ucoeff;
i=1;
while (norm((ucoeff-ucoeffp)/dt)>1e-9 || i<10) && i<10000
    ucoeffp=ucoeff;
    adv_coeff=Madv*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;


    rhs_coeff = -2/dt*ucoeff-lap_coeff+2*(adv_coeff);
    ucoeff = helmholtz_coeff(rhs_coeff, -2/dt, n, m);
    i=i+1;

end
disp([num2str(i) '    ' num2str(norm((ucoeff-ucoeffp)/dt))]);
g=ucoeff;
end
function ucoeff=time_relaxed_Linv_f(Mvor,Mgyro,Mlap,S,forcing,Mint,MintSq,n,m,dt)

ucoeff=zeros(n*m,1);
    
Madv=S*(Mvor)+Mgyro;

%% Loop!
ucoeffp=zeros(n*m,1);
% adv_coeff=Madv*ucoeff;
% lap_coeff=Mlap*ucoeff;
i=1;
while (norm((ucoeff-ucoeffp)/dt)>1e-9 || i<20) && i<10000
    ucoeffp=ucoeff;
    adv_coeff=Madv*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;


    rhs_coeff = -2/dt*ucoeff-lap_coeff+2*(adv_coeff-forcing);
    ucoeff = helmholtz_coeff(rhs_coeff, -2/dt, n, m);
    i=i+1;

end
disp([num2str(i) '    ' num2str(norm((ucoeff-ucoeffp)/dt)) '    ' num2str(abs(Mint*ucoeff*2*pi))]);
end