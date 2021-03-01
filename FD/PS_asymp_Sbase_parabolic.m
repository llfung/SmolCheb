Pef_array=[0.1:0.1:1 1.25:0.25:4 4.1:0.1:16 16.25:0.25:20 20.5:0.5:32 34:2:128 132:4:256 2.^(9:14)];
Vc1_array=NaN(4,length(Pef_array));

f=figure;
a=gca;a.YScale='log';a.XScale='log';
col_ind=a.ColorOrder;
a.ColorOrder=col_ind(1:5,:);
a.FontSize=14;
hold on;

for i=1:length(Pef_array)
Pef=Pef_array(i);
load(['D:\db\Smol\Asymp\parabolic\beta0\Asymp_para_beta_0B_0.31Pef_' num2str(Pef) '.mat'],'Vc1');
Vc1_array(1,i)=Vc1(1);
Vc1_array(2,i)=Vc1(length(Vc1)/8+1);
Vc1_array(3,i)=Vc1(length(Vc1)/4+1);
Vc1_array(4,i)=Vc1(length(Vc1)/8*3+1);
end

plot(Pef_array,abs(Vc1_array));
plot(10.^(-1:0.1:6),75*(10.^(-1:0.1:6)).^(-2),'k-','LineWidth',1);

for i=1:length(Pef_array)
Pef=Pef_array(i);
load(['D:\db\Smol\Asymp\parabolic\beta0.21\Asymp_para_beta_0.21B_0.31Pef_' num2str(Pef) '.mat'],'Vc1');
Vc1_array(1,i)=Vc1(1);
Vc1_array(2,i)=Vc1(length(Vc1)/8+1);
Vc1_array(3,i)=Vc1(length(Vc1)/4+1);
Vc1_array(4,i)=Vc1(length(Vc1)/8*3+1);
end
plot(Pef_array,abs(Vc1_array),'-.');
hold on;
ylim([1e-7 1e-1 ]);
xlim([1e-1 1e4]);
xlabel('$$Pe_f$$','Interpreter','latex');
ylabel('$$|V_{x,g,c}|$$','Interpreter','latex');
legend('$$z=-1$$','$$z=-0.75$$','$$z=-0.5$$','$$z=-0.25$$','$$Pe_f^{-2}$$','Interpreter','latex');

saveas(f,'Vcx_Pefvar_beta0_0.21.fig');
saveas(f,'Vcx_Pefvar_beta0_0.21.eps','epsc');

f=figure;
a=gca;a.YScale='log';a.XScale='log';
col_ind=a.ColorOrder;
a.ColorOrder=col_ind(1:4,:);
a.FontSize=14;
hold on;

for i=1:length(Pef_array)
Pef=Pef_array(i);
load(['D:\db\Smol\Asymp\parabolic\beta2.2\Asymp_para_beta_2.2B_0.31Pef_' num2str(Pef) '.mat'],'Vc1');
Vc1_array(1,i)=Vc1(1);
Vc1_array(2,i)=Vc1(length(Vc1)/8+1);
Vc1_array(3,i)=Vc1(length(Vc1)/4+1);
Vc1_array(4,i)=Vc1(length(Vc1)/8*3+1);
end
plot(Pef_array,abs(Vc1_array),'-');
hold on;plot(10.^(-1:0.1:6),75*(10.^(-1:0.1:6)).^(-2),'k-','LineWidth',1);
ylim([1e-7 1e-1 ]);
xlim([1e-1 1e4]);
xlabel('$$Pe_f$$','Interpreter','latex');
ylabel('$$|V_{x,g,c}|$$','Interpreter','latex');
legend('$$z=-1$$','$$z=-0.75$$','$$z=-0.5$$','$$z=-0.25$$','$$Pe_f^{-2}$$','Interpreter','latex');

saveas(f,'Vcx_Pefvar_beta2.2.fig');
saveas(f,'Vcx_Pefvar_beta2.2.eps','epsc');

%%

Pef_array=[2 4 8 16 32 64 128 256 512 1024];
legendCell = strcat('$$Pe_f=',string(num2cell(Pef_array)),'$$');
legendCell{length(Pef_array)+1}='$$S^{-3}$$';

f=figure;
cmap=colormap('parula');
cend=cmap(end,:);
cmap=cmap(1:ceil(size(cmap,1)/length(Pef_array)):end,:);
cmap=[cmap(end:-1:1,:);cend]; 

a=gca;a.YScale='log';a.XScale='log';
a.ColorOrder=cmap;a.FontSize=14;
hold on;
for i=1:length(Pef_array)
Pef=Pef_array(i);
load(['D:\db\Smol\Asymp\parabolic\beta0\Asymp_para_beta_0B_0.31Pef_' num2str(Pef) '.mat'],'Vc1','S_loop');
plot(abs(S_loop),abs(Vc1));
% plot(S_loop,Vc1);
end

xlim([1.25e-1 1e3]);ylim([1e-6 1e2]);
xlabel('$$|S(x)|$$','Interpreter','latex');ylabel('$$|V_{x,g,c}|$$','Interpreter','latex');
% legend(legendCell,'Interpreter','latex','FontSize',10,'NumColumns',1);
plot(10.^(-1:1:3),2e3*(10.^(-1:1:3)).^(-3),'k-','LineWidth',1);
% saveas(f,'Vcx_Svar_beta0.fig');
% saveas(f,'Vcx_Svar_beta0.eps','epsc');

% f=figure;a=gca;a.YScale='log';a.XScale='log';hold on;
% a.ColorOrder=cmap;a.FontSize=14;
for i=1:length(Pef_array)
Pef=Pef_array(i);
load(['D:\db\Smol\Asymp\parabolic\beta0.21\Asymp_para_beta_0.21B_0.31Pef_' num2str(Pef) '.mat'],'Vc1','S_loop');
plot(abs(S_loop),abs(Vc1),'-.');
% plot(S_loop,Vc1);
end

xlim([1.25e-1 1e3]);ylim([1e-6 1e2]);
xlabel('$$|S(x)|$$','Interpreter','latex');ylabel('$$|V_{x,g,c}|$$','Interpreter','latex');
legend(legendCell,'Interpreter','latex','FontSize',10,'NumColumns',1);

saveas(f,'Vcx_Svar_beta0_0.21.fig');
saveas(f,'Vcx_Svar_beta0_0.21.eps','epsc');

f=figure;
a=gca;a.YScale='log';a.XScale='log';hold on;
a.ColorOrder=cmap;a.FontSize=14;
for i=1:length(Pef_array)
Pef=Pef_array(i);
load(['D:\db\Smol\Asymp\parabolic\beta2.2\Asymp_para_beta_2.2B_0.31Pef_' num2str(Pef) '.mat'],'Vc1','S_loop');
plot(abs(S_loop),abs(Vc1));
% plot(S_loop,Vc1);
end
plot(10.^(-1:1:3),2e3*(10.^(-1:1:3)).^(-3),'k-','LineWidth',1);
xlim([1.25e-1 1e3]);ylim([1e-6 1e2]);
xlabel('$$|S(x)|$$','Interpreter','latex');ylabel('$$|V_{x,g,c}|$$','Interpreter','latex');
legend(legendCell,'Interpreter','latex','FontSize',10,'NumColumns',1);

saveas(f,'Vcx_Svar_beta2.2.fig');
saveas(f,'Vcx_Svar_beta2.2.eps','epsc');

%%
Pef_array=[2 8 32 128 512];
legendCell = strcat('$$Pe_f=',string(num2cell(Pef_array)),'$$');
% legendCell{length(Pef_array)+1}='$$S^{-3}$$';

f=figure;
cmap=colormap('parula');
cend=cmap(end,:);
cmap=cmap(1:ceil(size(cmap,1)/length(Pef_array)):end,:);
cmap=[cmap(end:-1:1,:);cend]; 
a=gca;a.YScale='log';a.XScale='log';hold on;
a.ColorOrder=cmap;a.FontSize=14;

for i=1:length(Pef_array)
Pef=Pef_array(i);
load(['D:\db\Smol\Asymp\parabolic\beta0\Asymp_para_beta_0B_0.31Pef_' num2str(Pef) '.mat'],'rese_array','S_loop');
plot(abs(S_loop),abs(rese_array(:,1))');
% plot(S_loop,Vc1);
end
% 
% for i=length(Pef_array):-1:1
% Pef=Pef_array(i);
% load(['D:\db\Smol\Asymp\parabolic\beta0.21\Asymp_para_beta_0.21B_0.31Pef_' num2str(Pef) '.mat'],'rese_array','S_loop');
% plot(abs(S_loop),abs(rese_array(:,1))','-.');
% % plot(S_loop,Vc1);
% end
plot(10.^(-1:1:3),5*(10.^(-1:1:3)).^(-2),'k-','LineWidth',1);

for i=1:length(Pef_array)
Pef=Pef_array(i);
load(['D:\db\Smol\Asymp\parabolic\beta2.2\Asymp_para_beta_2.2B_0.31Pef_' num2str(Pef) '.mat'],'rese_array','S_loop');
plot(abs(S_loop),abs(rese_array(:,1))','--');
% plot(S_loop,Vc1);
end

xlim([1.25e-1 1e3]);ylim([1e-6 1]);
legend('$$Pe_f=2, \beta=0$$','$$Pe_f=8, \beta=0$$','$$Pe_f=32, \beta=0$$','$$Pe_f=128, \beta=0$$','$$Pe_f=512, \beta=0$$','$$Pe_f^{-2}$$',...
    '$$Pe_f=2, \beta=2.2$$','$$Pe_f=8, \beta=2.2$$','$$Pe_f=32, \beta=2.2$$','$$Pe_f=128, \beta=2.2$$','$$Pe_f=512, \beta=2.2$$',...
    'Interpreter','latex','NumColumns',2,'location','southwest','FontSize',10);

xlabel('$$|S(x)|$$','Interpreter','latex');ylabel('$$|D_{xx,g,c}|$$','Interpreter','latex');

% Create textarrow
annotation(f,'textarrow',[0.230357142857143 0.269642857142857],...
    [0.873809523809524 0.826190476190476],'String',{'$$\beta=0$$'},...
    'Interpreter','latex');

% Create textarrow
annotation(f,'textarrow',[0.230357142857143 0.266071428571429],...
    [0.740476190476191 0.778571428571429],'String',{'$$\beta=2.2$$'},...
    'Interpreter','latex');

saveas(f,'Dxx_Svar_beta0_2.2.fig');
saveas(f,'Dxx_Svar_beta0_2.2.eps','epsc');

%%
beta_array=[0:0.2:2.2];
legendCell = strcat('$$\beta=',string(num2cell(beta_array)),'$$');

f=figure;
cmap=colormap('parula');
cend=cmap(end,:);
cmap=cmap(1:ceil(size(cmap,1)/length(beta_array)):end,:);
cmap=[cmap(end:-1:1,:);cend]; 
a=gca;hold on;
a.ColorOrder=cmap;a.FontSize=14;

for i=1:length(beta_array)
beta=beta_array(i);
load(['D:\db\Smol\Asymp\parabolic\beta_var\Asymp_para_beta_' num2str(beta) 'B_0.31Pef_64.mat'],'Vc1','S_loop');
plot(S_loop,Vc1);
end
xlim([-20 20]);
xlabel('$$S(x)$$','Interpreter','latex');ylabel('$$V_{x,g,c}$$','Interpreter','latex');
legend(legendCell,'Interpreter','latex','location','southwest','FontSize',10,'NumColumns',2);

saveas(f,'Vcx_betavar_Pef64.fig');
saveas(f,'Vcx_betavar_Pef64.eps','epsc');
