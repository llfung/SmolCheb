name='0.21beta_0.31B';

% load('E:\db\Smol\bearon2011\smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_0.5Pef_dx_448dz_448_m8_n8_dt0.001_tf35.002\smol_pBC_bearon2011_2.2beta_0.31B_0.25Vc_0DT_0.5Pef_dx_448dz_448_m8_n8_dt0.001_tf35.002_PS.mat')
% load('E:\db\Smol\bearon2011\smol_pBC_bearon2011_2.2beta_0B_0.25Vc_0DT_0.5Pef_dx_384dz_384_m8_n12_dt0.0025_tf30.005_combined.mat');
% load('E:\db\Smol\bearon2011\smol_pBC_bearon2011_0.21beta_0B_0.25Vc_0DT_0.5Pef_dx_256dz_256_m8_n8_dt0.01_tf25.02_combined.mat');
load('E:\db\Smol\bearon2011\smol_pBC_bearon2011_0.21beta_0.31B_0.25Vc_0DT_0.5Pef_dx_384dz_384_m8_n16_dt0.002_tf75.004_PS.mat')
%% Density plot
f=figure('Position',[0,564,350,332]);
[C,h]=contour(x,z,reshape(real(cell_den(end,:)),Nx_mesh,Nz_mesh)',0.05:0.05:0.5);
% [C,h]=contour(x,z,reshape(real(cell_den(end,:)),Nx_mesh,Nz_mesh)',0.22:0.005:0.265);
% contourf(x,z,reshape(real(cell_den(end,:)),Nx_mesh,Nz_mesh)',0.22:0.005:0.265,'ShowText','on','LabelSpacing',200);
xlabel('$$x$$','Interpreter','latex','FontSize',12,'Position',[0,-1.1,0]);ylabel('$$z$$','Interpreter','latex','FontSize',12,'Position',[-1.1,0,0]);
axis equal; axis([-1 1 -1 1]);set(gca,'FontSize',12)
%clabel(C,h,'manual');
% saveas(f,['./figs/2D' name '_nfs.fig']);
% saveas(f,['./figs/2D' name '_nfs.eps'],'epsc');

f=figure('Position',[0,564,350,332]);
[C,h]=contour(x,z,reshape(real(ng(end,:)),Nx_mesh,Nz_mesh)',0.05:0.05:0.5);
% [C,h]=contour(x,z,reshape(real(cell_den(end,:)),Nx_mesh,Nz_mesh)',0.22:0.005:0.265);
% contourf(x,z,reshape(real(ng(end,:)),Nx_mesh,Nz_mesh)',0.22:0.005:0.265,'ShowText','on','LabelSpacing',200);
xlabel('$$x$$','Interpreter','latex','FontSize',12,'Position',[0,-1.1,0]);ylabel('$$z$$','Interpreter','latex','FontSize',12,'Position',[-1.1,0,0]);
axis equal; axis([-1 1 -1 1]);set(gca,'FontSize',12)
%clabel(C,h,'manual');
% saveas(f,['./figs/2D' name '_ngs.fig']);
% saveas(f,['./figs/2D' name '_ngs.eps'],'epsc');

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

%% Calculating drifts and dispersions
% f=ucoeff./real(Mint*ucoeff)/2/pi;
% ex=real(Mint*Mp1*ucoeff)./real(Mint*ucoeff);
% ez=real(Mint*Mp3*ucoeff)./real(Mint*ucoeff);
% zero_row=zeros(1,Nx_mesh*Nz_mesh);k=n*m/2+m/2+1;
% 
% bx_RHS=Mp1*f-ex.*f;
% bz_RHS=Mp3*f-ez.*f;
%         
% inhomo_RHS=Mp1*(f*Rdx)+Mp3*(f*Rdz)-(ex*Rdx+ez*Rdz).*f;
% 
% fu_RHS=U_profile.*(f*Rdx)+W_profile.*(f*Rdz);
% 
% bx_RHS=Msin*bx_RHS;bx_RHS(k,:)=zero_row;
% bz_RHS=Msin*bz_RHS;bz_RHS(k,:)=zero_row;
% inhomo_RHS=Msin*inhomo_RHS;inhomo_RHS(k,:)=zero_row;
% fu_RHS=Msin*fu_RHS;fu_RHS(k,:)=zero_row;
% 
% nm=n*m;N_mesh=Nx_mesh*Nz_mesh;
% bx=NaN(nm,N_mesh);
% bz=NaN(nm,N_mesh);
% f_inhomo=NaN(nm,N_mesh);
% fu=NaN(nm,N_mesh);
% 
% for j=1:N_mesh
%     Le=full(Msin*gather(curl_profile(j)*(Mvor)+E_profile(j)*(Mstrain)+Mgyro-Mlap));
%     Le(n*m/2+m/2+1,:)=Mint;
%     
%     LHS=Le\[bx_RHS(:,j) bz_RHS(:,j) inhomo_RHS(:,j) fu_RHS(:,j)];
% 
%     bx(:,j)=LHS(:,1);
%     bz(:,j)=LHS(:,2);
%     f_inhomo(:,j)=LHS(:,3);
%     fu(:,j)=LHS(:,4);
% end
% clearvars Le LHS zero_row bx_RHS bz_RHS inhomo_RHS fu_RHS adv_coeff;
%   
%         Dxxf=real(Mint*(Mp1*reshape(bx,nm,N_mesh))*(2*pi));
%         Dxzf=real(Mint*(Mp1*reshape(bz,nm,N_mesh))*(2*pi));
%         Dzxf=real(Mint*(Mp3*reshape(bx,nm,N_mesh))*(2*pi));
%         Dzzf=real(Mint*(Mp3*reshape(bz,nm,N_mesh))*(2*pi));
%         Vixf=real(Mint*(Mp1*reshape(f_inhomo,nm,N_mesh))*(2*pi));
%         Vizf=real(Mint*(Mp3*reshape(f_inhomo,nm,N_mesh))*(2*pi));
%         
%         Vuxf=real(Mint*(Mp1*reshape(fu,nm,N_mesh))*(2*pi));
%         Vuzf=real(Mint*(Mp3*reshape(fu,nm,N_mesh))*(2*pi));
% 
% clearvars bx bz f_inhomo fu;

%% Prepare drifts and dispersion for plots
pfx=real(reshape(real(Mint*Mp1*ucoeff)./real(Mint*ucoeff),Nx_mesh,Nz_mesh))';
pfz=real(reshape(real(Mint*Mp3*ucoeff)./real(Mint*ucoeff),Nx_mesh,Nz_mesh))';

ex_avg_plt=reshape(ex_avg,Nx_mesh,Nz_mesh)';
ez_avg_plt=reshape(ez_avg,Nx_mesh,Nz_mesh)';

Vix_plt=reshape(Vix,Nx_mesh,Nz_mesh)';
Viz_plt=reshape(Viz,Nx_mesh,Nz_mesh)';
Vux_plt=reshape(Vux,Nx_mesh,Nz_mesh)';
Vuz_plt=reshape(Vuz,Nx_mesh,Nz_mesh)';
Dx_plt=reshape(Dxx*Rdx+Dzx*Rdz,Nx_mesh,Nz_mesh)';
Dz_plt=reshape(Dxz*Rdx+Dzz*Rdz,Nx_mesh,Nz_mesh)';

Vixf_plt=reshape(Vixf,Nx_mesh,Nz_mesh)';
Vizf_plt=reshape(Vizf,Nx_mesh,Nz_mesh)';
Vuxf_plt=reshape(Vuxf,Nx_mesh,Nz_mesh)';
Vuzf_plt=reshape(Vuzf,Nx_mesh,Nz_mesh)';
Dxf_plt=reshape(Dxxf*Rdx+Dzxf*Rdz,Nx_mesh,Nz_mesh)';
Dzf_plt=reshape(Dxzf*Rdx+Dzzf*Rdz,Nx_mesh,Nz_mesh)';
Dxxf_plt=reshape(Dxxf,Nx_mesh,Nz_mesh)';
Dxzf_plt=reshape(Dxzf,Nx_mesh,Nz_mesh)';
Dzxf_plt=reshape(Dzxf,Nx_mesh,Nz_mesh)';
Dzzf_plt=reshape(Dzzf,Nx_mesh,Nz_mesh)';

Uplot=reshape(U_profile,Nx_mesh,Nz_mesh)';
Wplot=reshape(W_profile,Nx_mesh,Nz_mesh)';

Uppfx=reshape(U_profile,Nx_mesh,Nz_mesh)'+Vc*pfx;
Wppfz=reshape(W_profile,Nx_mesh,Nz_mesh)'+Vc*pfz;

adv_x=Uplot+Vc*ex_avg_plt-Vc^2*Vix_plt-Vc*Vux_plt-Vc^2*Dx_plt;
adv_z=Wplot+Vc*ez_avg_plt-Vc^2*Viz_plt-Vc*Vuz_plt-Vc^2*Dz_plt;

advf_x=Uplot+Vc*pfx-Vc^2*Vixf_plt-Vc*Vuxf_plt-Vc^2*Dxf_plt;
advf_z=Wplot+Vc*pfz-Vc^2*Vizf_plt-Vc*Vuzf_plt-Vc^2*Dzf_plt;

nfs=reshape(real(cell_den(end,:)),Nx_mesh,Nz_mesh)';
nfs_dx=reshape(real(cell_den(end,:)*Rdx),Nx_mesh,Nz_mesh)';
nfs_dz=reshape(real(cell_den(end,:)*Rdz),Nx_mesh,Nz_mesh)';
ex_est=Vc*ex_avg_plt-Vc^2*Vixf_plt-Vc*Vuxf_plt-Vc^2*(Dxxf_plt.*nfs_dx+Dxzf_plt.*nfs_dz)./nfs;
ez_est=Vc*ez_avg_plt-Vc^2*Vizf_plt-Vc*Vuzf_plt-Vc^2*(Dzxf_plt.*nfs_dx+Dzzf_plt.*nfs_dz)./nfs;

% Dx_flux=(reshape(Dxx,Nx_mesh,Nz_mesh)'.*nfs_dx+reshape(Dxz,Nx_mesh,Nz_mesh)'.*nfs_dz)./nfs;
% Dz_flux=(reshape(Dzx,Nx_mesh,Nz_mesh)'.*nfs_dx+reshape(Dzz,Nx_mesh,Nz_mesh)'.*nfs_dz)./nfs;
% Dxf_flux=(Dxxf_plt.*nfs_dx+Dxzf_plt.*nfs_dz)./nfs;
% Dzf_flux=(Dzxf_plt.*nfs_dx+Dzzf_plt.*nfs_dz)./nfs;
%% Plots
f=figure('Position',[0,200,463,383]);
contourf(x,z,sqrt(ex_avg_plt.^2+ez_avg_plt.^2),100,'LineStyle','none');
c=1-winter*0.5;
colormap(c);colorbar('eastoutside');
hold on;
quiver(x(end/32:end/16:end),z(end/32:end/16:end),ex_avg_plt(end/32:end/16:end,end/32:end/16:end),ez_avg_plt(end/32:end/16:end,end/32:end/16:end),'Color','k','AutoScaleFactor',0.5);
hold off;
xlabel('$$x$$','Interpreter','latex','FontSize',16,'Position',[0,-1.15,0]);ylabel('$$z$$','Interpreter','latex','FontSize',16,'Position',[-1.15,0,0]);
set(gca,'FontSize',16);
axis equal; axis([-1 1 -1 1]);
saveas(f,['./figs/2D' name '_eg.fig']);
saveas(f,['./figs/2D' name '_eg.eps'],'epsc');

f=figure('Position',[0,200,463,383]);
contourf(x,z,Vc*sqrt(Vix_plt.^2+Viz_plt.^2),100,'LineStyle','none');
c=1-winter*0.5;
colormap(c);colorbar('eastoutside');
hold on;
quiver(x(end/32:end/16:end),z(end/32:end/16:end),Vc*Vix_plt(end/32:end/16:end,end/32:end/16:end),Vc*Viz_plt(end/32:end/16:end,end/32:end/16:end),'Color','k','AutoScaleFactor',0.5);
hold off;
xlabel('$$x$$','Interpreter','latex','FontSize',16,'Position',[0,-1.15,0]);ylabel('$$z$$','Interpreter','latex','FontSize',16,'Position',[-1.15,0,0]);
set(gca,'FontSize',16);
axis equal; axis([-1 1 -1 1]);
saveas(f,['./figs/2D' name '_Vig.fig']);
saveas(f,['./figs/2D' name '_Vig.eps'],'epsc');

f=figure('Position',[0,200,463,383]);
contourf(x,z,Pef*sqrt(Vux_plt.^2+Vuz_plt.^2),100,'LineStyle','none');
c=1-winter*0.5;
colormap(c);colorbar('eastoutside');
hold on;
quiver(x(end/32:end/16:end),z(end/32:end/16:end),Pef*Vux_plt(end/32:end/16:end,end/32:end/16:end),Pef*Vuz_plt(end/32:end/16:end,end/32:end/16:end),'Color','k','AutoScaleFactor',0.5);
hold off;
xlabel('$$x$$','Interpreter','latex','FontSize',16,'Position',[0,-1.15,0]);ylabel('$$z$$','Interpreter','latex','FontSize',16,'Position',[-1.15,0,0]);
set(gca,'FontSize',16);
axis equal; axis([-1 1 -1 1]);
saveas(f,['./figs/2D' name '_Vug.fig']);
saveas(f,['./figs/2D' name '_Vug.eps'],'epsc');


% f=figure('Position',[0,200,400,500]);
% contourf(x,z,Vc^2*sqrt(Uplot.^2+Wplot.^2),100,'LineStyle','none');
% c=1-winter*0.5;
% colormap(c);colorbar('southoutside');
% hold on;
% quiver(x(end/32:end/16:end),z(end/32:end/16:end),Vc^2*Vux_plt(end/32:end/16:end,end/32:end/16:end),Vc^2*Vuz_plt(end/32:end/16:end,end/32:end/16:end),'Color','k','AutoScaleFactor',0.5);
% hold off;
% xlabel('$$x$$','Interpreter','latex','FontSize',12);ylabel('$$z$$','Interpreter','latex','FontSize',12);
% axis equal; axis([-1 1 -1 1]);

f=figure('Position',[0,200,463,383]);
contourf(x,z,sqrt(pfx.^2+pfz.^2),100,'LineStyle','none');
c=1-winter*0.5;
colormap(c);colorbar('eastoutside');
hold on;
quiver(x(end/32:end/16:end),z(end/32:end/16:end),pfx(end/32:end/16:end,end/32:end/16:end),pfz(end/32:end/16:end,end/32:end/16:end),'Color','k','AutoScaleFactor',0.5);
hold off;
xlabel('$$x$$','Interpreter','latex','FontSize',16,'Position',[0,-1.15,0]);ylabel('$$z$$','Interpreter','latex','FontSize',16,'Position',[-1.15,0,0]);
set(gca,'FontSize',16);
axis equal; axis([-1 1 -1 1]);
saveas(f,['./figs/2D' name '_ef.fig']);
saveas(f,['./figs/2D' name '_ef.eps'],'epsc');

f=figure('Position',[0,200,463,383]);
contourf(x,z,Vc*sqrt(Vixf_plt.^2+Vizf_plt.^2),100,'LineStyle','none');
c=1-winter*0.5;
colormap(c);colorbar('eastoutside');
hold on;
quiver(x(end/32:end/16:end),z(end/32:end/16:end),Vc*Vixf_plt(end/32:end/16:end,end/32:end/16:end),Vc*Vizf_plt(end/32:end/16:end,end/32:end/16:end),'Color','k','AutoScaleFactor',0.5);
hold off;
xlabel('$$x$$','Interpreter','latex','FontSize',16,'Position',[0,-1.15,0]);ylabel('$$z$$','Interpreter','latex','FontSize',16,'Position',[-1.15,0,0]);
set(gca,'FontSize',16);
axis equal; axis([-1 1 -1 1]);
saveas(f,['./figs/2D' name '_Vif.fig']);
saveas(f,['./figs/2D' name '_Vif.eps'],'epsc');

f=figure('Position',[0,200,463,383]);
contourf(x,z,sqrt(Vuxf_plt.^2+Vuzf_plt.^2),100,'LineStyle','none');
c=1-winter*0.5;
colormap(c);colorbar('eastoutside');
hold on;
quiver(x(end/32:end/16:end),z(end/32:end/16:end),Vuxf_plt(end/32:end/16:end,end/32:end/16:end),Vuzf_plt(end/32:end/16:end,end/32:end/16:end),'Color','k','AutoScaleFactor',0.5);
hold off;
xlabel('$$x$$','Interpreter','latex','FontSize',16,'Position',[0,-1.15,0]);ylabel('$$z$$','Interpreter','latex','FontSize',16,'Position',[-1.15,0,0]);
set(gca,'FontSize',16)
axis equal; axis([-1 1 -1 1]);
saveas(f,['./figs/2D' name '_Vuf.fig']);
saveas(f,['./figs/2D' name '_Vuf.eps'],'epsc');

f=figure('Position',[0,200,463,383]);
contourf(x,z,Vc*sqrt((Vixf_plt-Vix_plt).^2+(Vizf_plt-Viz_plt).^2),100,'LineStyle','none');
c=1-winter*0.5;
colormap(c);colorbar('eastoutside');
hold on;
quiver(x(end/32:end/16:end),z(end/32:end/16:end),Vc*(Vixf_plt(end/32:end/16:end,end/32:end/16:end)-Vix_plt(end/32:end/16:end,end/32:end/16:end)),Vc*(Vizf_plt(end/32:end/16:end,end/32:end/16:end)-Viz_plt(end/32:end/16:end,end/32:end/16:end)),'Color','k','AutoScaleFactor',0.5);
hold off;
xlabel('$$x$$','Interpreter','latex','FontSize',16,'Position',[0,-1.15,0]);ylabel('$$z$$','Interpreter','latex','FontSize',16,'Position',[-1.15,0,0]);
set(gca,'FontSize',16);
axis equal; axis([-1 1 -1 1]);
saveas(f,['./figs/2D' name '_Vif-g.fig']);
saveas(f,['./figs/2D' name '_Vif-g.eps'],'epsc');

f=figure('Position',[0,200,463,383]);
contourf(x,z,sqrt((Vuxf_plt-Vux_plt).^2+(Vuzf_plt-Vuz_plt).^2),100,'LineStyle','none');
c=1-winter*0.5;
colormap(c);colorbar('eastoutside');
hold on;
quiver(x(end/32:end/16:end),z(end/32:end/16:end),(Vuxf_plt(end/32:end/16:end,end/32:end/16:end)-Vux_plt(end/32:end/16:end,end/32:end/16:end)),(Vuzf_plt(end/32:end/16:end,end/32:end/16:end)-Vuz_plt(end/32:end/16:end,end/32:end/16:end)),'Color','k','AutoScaleFactor',0.5);
hold off;
xlabel('$$x$$','Interpreter','latex','FontSize',16,'Position',[0,-1.15,0]);ylabel('$$z$$','Interpreter','latex','FontSize',16,'Position',[-1.15,0,0]);
set(gca,'FontSize',16)
axis equal; axis([-1 1 -1 1]);
saveas(f,['./figs/2D' name '_Vuf-g.fig']);
saveas(f,['./figs/2D' name '_Vuf-g.eps'],'epsc');

%%
% subplot(2,3,2);
% contour(x,z,Vc^2*sqrt(Vix_plt.^2+Viz_plt.^2));colorbar;
% subplot(2,3,3);
% contour(x,z,Vc*sqrt(Vux_plt.^2+Vuz_plt.^2));colorbar;
% subplot(2,3,4);
% contour(x,z,Vc^2*sqrt(Dx_plt.^2+Dz_plt.^2));colorbar;
% subplot(2,3,5);
% contour(x,z,sqrt(Uplot.^2+Wplot.^2));colorbar;
% subplot(2,3,6);
% contour(x,z,sqrt(adv_x.^2+adv_z.^2));colorbar;

% figure;
% subplot(2,3,1);
% contour(x,z,Vc*sqrt(pfx.^2+pfz.^2));colorbar;
% subplot(2,3,2);
% contour(x,z,Vc^2*sqrt(Vixf_plt.^2+Vizf_plt.^2));colorbar;
% subplot(2,3,3);
% contour(x,z,Vc*sqrt(Vuxf_plt.^2+Vuzf_plt.^2));colorbar;
% subplot(2,3,4);
% contour(x,z,Vc^2*sqrt(Dxf_plt.^2+Dzf_plt.^2));colorbar;
% subplot(2,3,5);
% contour(x,z,sqrt(Uplot.^2+Wplot.^2));colorbar;
% subplot(2,3,6);
% contour(x,z,sqrt(advf_x.^2+advf_z.^2));colorbar;

%% 
f=figure('Position',[618,624,240,291]);
contour(x,z,reshape(Vc*Dxx,Nx_mesh,Nz_mesh)');
colorbar('northoutside');
axis equal;axis([-1,1,-1,1]);
%title('$$Pe_s D_{xx,g,c}$$','Interpreter','latex');
ylabel('$$z$$','Interpreter','latex');
xlabel('$$x$$','Interpreter','latex');
set(gca,'FontSize',12);
saveas(f,['./figs/2D' name '_Dxxg.fig']);
saveas(f,['./figs/2D' name '_Dxxg.eps'],'epsc');

f=figure('Position',[680,644,222,291]);
contour(x,z,reshape(Vc*Dxz,Nx_mesh,Nz_mesh)');
colorbar('northoutside');
axis equal;axis([-1,1,-1,1]);
%title('$$Pe_s D_{xz,g,c}$$','Interpreter','latex');
xlabel('$$x$$','Interpreter','latex');
set(gca,'FontSize',12);
saveas(f,['./figs/2D' name '_Dxzg.fig']);
saveas(f,['./figs/2D' name '_Dxzg.eps'],'epsc');

f=figure('Position',[680,644,222,291]);
contour(x,z,reshape(Vc*Dzx,Nx_mesh,Nz_mesh)');
colorbar('northoutside');
axis equal;axis([-1,1,-1,1]);
%title('$$Pe_s D_{zx,g,c}$$','Interpreter','latex');
xlabel('$$x$$','Interpreter','latex');
%ylabel('$$z$$','Interpreter','latex');
set(gca,'FontSize',12);
saveas(f,['./figs/2D' name '_Dzxg.fig']);
saveas(f,['./figs/2D' name '_Dzxg.eps'],'epsc');

f=figure('Position',[680,644,222,291]);
contour(x,z,reshape(Vc*Dzz,Nx_mesh,Nz_mesh)');
colorbar('northoutside');
axis equal;axis([-1,1,-1,1]);
%title('$$Pe_s D_{zz,g,c}$$','Interpreter','latex');
xlabel('$$x$$','Interpreter','latex');
set(gca,'FontSize',12);
saveas(f,['./figs/2D' name '_Dzzg.fig']);
saveas(f,['./figs/2D' name '_Dzzg.eps'],'epsc');

f=figure('Position',[680,644,240,291]);
contour(x,z,reshape(Vc*Dxxf,Nx_mesh,Nz_mesh)');
colorbar('southoutside');
axis equal;axis([-1,1,-1,1]);
%title('$$D_{xx,c}$$','Interpreter','latex');
ylabel('$$z$$','Interpreter','latex');
set(gca,'FontSize',12);
saveas(f,['./figs/2D' name '_Dxxf.fig']);
saveas(f,['./figs/2D' name '_Dxxf.eps'],'epsc');

f=figure('Position',[680,644,222,291]);
%title('$$D_{xz,c}$$','Interpreter','latex');
contour(x,z,reshape(Vc*Dxzf,Nx_mesh,Nz_mesh)');
colorbar('southoutside');
axis equal;axis([-1,1,-1,1]);
set(gca,'FontSize',12);
saveas(f,['./figs/2D' name '_Dxzf.fig']);
saveas(f,['./figs/2D' name '_Dxzf.eps'],'epsc');

f=figure('Position',[680,644,222,291]);
%title('$$D_{zx,c}$$','Interpreter','latex');
contour(x,z,reshape(Vc*Dzxf,Nx_mesh,Nz_mesh)');
colorbar('southoutside');
axis equal;axis([-1,1,-1,1]);
%xlabel('$$x$$','Interpreter','latex');
%ylabel('$$z$$','Interpreter','latex');
set(gca,'FontSize',12);
saveas(f,['./figs/2D' name '_Dzxf.fig']);
saveas(f,['./figs/2D' name '_Dzxf.eps'],'epsc');

f=figure('Position',[680,644,222,291]);
contour(x,z,reshape(Vc*Dzzf,Nx_mesh,Nz_mesh)');
colorbar('southoutside');
axis equal;axis([-1,1,-1,1]);
%title('$$D_{zz,c}$$','Interpreter','latex');
%xlabel('$$x$$','Interpreter','latex');
%ylabel('$$z$$','Interpreter','latex');
set(gca,'FontSize',12);
saveas(f,['./figs/2D' name '_Dzzf.fig']);
saveas(f,['./figs/2D' name '_Dzzf.eps'],'epsc');