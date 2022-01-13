addpath(genpath('core'))
addpath(genpath('core/x_FD'))
addpath(genpath('core/p_DFS'))
addpath(genpath('core/PS_RT'))
% dir='D:\db\Smol\VS\DT\smol_pBC_2.2beta_0.31B_0Vsm_0Vsv_0.25Vc_0.05DT_1Pef_cospi_cd96_m16_n20_dt0.002_tf100.004\';

% load([dir 'Summary.mat']);
%%
[settings,Mvor,Mgyro,Mlap,Rd,Rd2,Mp1,Mp3,Mp1p3,~]=all_mat_gen(settings);
helm=helmholtz_genGPU( settings.n, settings.m);

%% Small Pe Cell Density
D=Vc*Transformed.Dxx_g-2*DT*Transformed.DDTxx_g+DT/Vc;
space_op=diag(Transformed.ex_g-Vc*Transformed.Vix_g+DT*Transformed.VDTx_g)*Rd-(Rd2*diag(D)+Rd*diag(D*Rd));
cell_den_smallPe=[zeros(1,N_mesh) 1/dx]/[space_op ones(N_mesh,1)];

D=Vc*Transformed.Dxx_g+DT/Vc;
space_op=diag(Transformed.ex_g)*Rd-(Rd2*diag(D)+Rd*diag(D*Rd));
cell_den_GTD=[zeros(1,N_mesh) 1/dx]/[space_op ones(N_mesh,1)];

% D=Vc*Transformed.Dzz_g-2*DT*Transformed.DDTzz_g+DT/Vc;
% space_op=diag(Transformed.ez_g-Vc*Transformed.Viz_g+DT*Transformed.VDTz_g)*Rd-(Rd2*diag(D)+Rd*diag(D*Rd));
% cell_den_smallPe=[zeros(1,N_mesh) 1/dz]/[space_op ones(N_mesh,1)];
% 
% D=Vc*Transformed.Dzz_g+DT/Vc;
% space_op=diag(Transformed.ez_g)*Rd-(Rd2*diag(D)+Rd*diag(D*Rd));
% cell_den_GTD=[zeros(1,N_mesh) 1/dz]/[space_op ones(N_mesh,1)];

f=figure;a=gca;
plot(x,cell_den(end,:),'k-',x,cell_den_smallPe,'b-.',x,cell_den_GTD,'r--');
% plot(z,cell_den(end,:),'k-',z,cell_den_smallPe,'b-.',z,cell_den_GTD,'r--');
a.FontSize=14;a.Position=[0.146428571428571,0.145238095238095,0.833928571428572,0.828571428571429];
xlabel('$$x$$','Interpreter','latex','FontSize',20);ylabel('$$n(x)$$','Interpreter','latex','FontSize',20);
% xlabel('$$z$$','Interpreter','latex','FontSize',20);ylabel('$$n(z)$$','Interpreter','latex','FontSize',20);
% legend('Exact Smol.','Asymp. Approx.','GTD','location','northeast','FontSize',11);
saveas(f,['beta' num2str(beta) '_B' num2str(B) '_Vc' num2str(Vc) '_Pef' num2str(Pef) '_DT' num2str(DT) '_nx.fig']);
saveas(f,['beta' num2str(beta) '_B' num2str(B) '_Vc' num2str(Vc) '_Pef' num2str(Pef) '_DT' num2str(DT) '_nx.eps'],'epsc');
% saveas(f,['beta' num2str(beta) '_B' num2str(B) '_Vc' num2str(Vc) '_Pef' num2str(Pef) '_DT' num2str(DT) '_nz.fig']);
% saveas(f,['beta' num2str(beta) '_B' num2str(B) '_Vc' num2str(Vc) '_Pef' num2str(Pef) '_DT' num2str(DT) '_nz.eps'],'epsc');

%%
f=figure;set(f,'defaultAxesColorOrder',[0 0 0;0 0 0]);
axes1 = axes('Parent',f);
hold on;

% yyaxis left;
plot(x,Transformed.ex(end,:),'b-');
plot(x,Transformed.ex_g,'b--');
% yyaxis right;
plot(x,Vc*Transformed.Dxx(end,:),'r-');
plot(x,Vc*Transformed.Dxx_g,'r--');
plot(x,Vc*Transformed.Vix(end,:),'g-');
plot(x,Vc*Transformed.Vix_g,'g--');
plot(x,-DT*Transformed.VDTx(end,:),'c-');
plot(x,-DT*Transformed.VDTx_g,'c--');
plot(x,-2*DT*Transformed.DDTxx(end,:),'m-');
plot(x,-2*DT*Transformed.DDTxx_g,'m--');

% % yyaxis left;
% plot(z,Transformed.ez(end,:),'b-');
% plot(z,Transformed.ez_g,'b--');
% % yyaxis right;
% plot(z,Vc*Transformed.Dzz(end,:),'r-');
% plot(z,Vc*Transformed.Dzz_g,'r--');
% plot(z,Vc*Transformed.Viz(end,:),'g-');
% plot(z,Vc*Transformed.Viz_g,'g--');

hold off;
set(f,'defaultAxesColorOrder',[1 1 1;1 1 1]);

% legend('$$\langle p_x \rangle_f$$','$$\langle p_x \rangle_g$$',...
%     '$$D_{xx,c}$$','$$Pe_s D_{xx,g,c}$$',...
%     '$$V_{x,c}$','$$Pe_s V_{x,g,c}$$',...
%     'Interpreter','latex','location','east','NumColumns',3,'FontSize',14);
% legend('$$\langle p_x \rangle_f$$','$$\langle p_x \rangle_g$$',...
%     '$$D_{xx,c}$$','$$Pe_s D_{xx,g,c}$$','$$V_{x,c}$','$$Pe_s V_{x,g,c}$$',...
%     '$$V_{x,D_T}$$','$$Pe_s V_{x,g,D_T}$$','$$D_{xx,D_T}$$','$$Pe_s D_{xx,g,D_T}$$',...
%     'Interpreter','latex','location','southoutside','NumColumns',5,'FontSize',10,...
% 'Position',[0.0434455807012359 0.0108730179343435 0.924411562155906 0.0869047596341088]);

% legend('$$\langle p_z \rangle_f$$','$$\langle p_z \rangle_g$$',...
%     '$$D_{zz,c}$$','$$Pe_s D_{zz,g,c}$$',...
%     '$$V_{z,c}$','$$Pe_s V_{z,g,c}$$',...
%     'Interpreter','latex','location','east','NumColumns',3,'FontSize',14);


axes1.FontSize=14;
xlabel('$$x$$','Interpreter','latex','FontSize',16);
ylabel('$$\langle p_x \rangle , D_{xx} , V_x$$','Interpreter','latex','FontSize',16);
% yyaxis right;ylabel('$$D_{xx} | V_x$$','Interpreter','latex','FontSize',16);
% yyaxis left;ylabel('$$\langle p_x \rangle$$','Interpreter','latex','FontSize',16);

% xlabel('$$z$$','Interpreter','latex','FontSize',16);
% ylabel('$$\langle p_z \rangle | D_{zz} | V_z$$','Interpreter','latex','FontSize',16);
% yyaxis right;ylabel('$$D_{zz} | V_z$$','Interpreter','latex','FontSize',16);
% yyaxis left;ylabel('$$\langle p_z \rangle$$','Interpreter','latex','FontSize',16);

saveas(f,['beta' num2str(beta) '_B' num2str(B) '_Vc' num2str(Vc) '_Pef' num2str(Pef) '_DT' num2str(DT) '_DVpavg.fig']);
saveas(f,['beta' num2str(beta) '_B' num2str(B) '_Vc' num2str(Vc) '_Pef' num2str(Pef) '_DT' num2str(DT) '_DVpavg.eps'],'epsc');