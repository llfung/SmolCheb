addpath(genpath('core'))
addpath(genpath('core/x_FD'))
addpath(genpath('core/p_DFS'))
addpath(genpath('core/PS_RT'))

%%
[settings,Mvor,Mgyro,Mlap,Rd,Rd2,Mp1,Mp3,Mp1p3,~]=all_mat_gen(settings);
Msin=kron(spdiags(0.5i*ones(settings.n,1)*[-1,1], [-1 1], settings.n, settings.n),speye(settings.m));
zero_row=zeros(1,N_mesh);
% helm=helmholtz_genGPU( settings.n, settings.m);
% helm.dt=settings.dt*1;

% e_FK=Transformed.ex_g;
% Vi_FK=Transformed.Vix_g;
% D_FK=Transformed.Dxx_g;


e_FK=Transformed.ez_g;
Vi_FK=Transformed.Viz_g;
D_FK=Transformed.Dzz_g;
x=z;dx=dz;
%% Dxx at t=0
f0=zeros(n*m,N_mesh);f0(m*n/2+m/2+1,:)=1/4/pi;

% b_f0=time_relaxed_Linv(gpuArray(Mvor),gpuArray(Mgyro),gpuArray(Mlap),gpuArray(S_profile),...
%         Mp1*f0,gpuArray(settings.Mint),gpuArray(settings.MintSq),helm);
%     
% ex_avg0=zeros(1,N_mesh);
% Vix0=zeros(1,N_mesh);
% Dxx0=real(settings.Mint*Mp1*b_f0*2*pi);
    
[~,Linv]=Linv_g(S_profile,Mvor,Mgyro,Mlap,settings.Mint,Msin,n,m);
% 
% [ex_avg0,ez_avg0,Dxx0,Dxz0,Dzx0,Dzz0,Vix0,Viz0]=...
%     Linv_f('x',f0,Linv,Msin,Rd,Rd2,Mp1,Mp3,settings,...
%     zero_row,settings.n*settings.m/2+settings.m/2+1);

[ex_avg0,ez_avg0,Dxx0,Dxz0,Dzx0,Dzz0,Vix0,Viz0]=...
    Linv_f('z',f0,Linv,Msin,Rd,Rd2,Mp1,Mp3,settings,...
    zero_row,settings.n*settings.m/2+settings.m/2+1);

% D0_dn_n=Vc*Dxx0.*(real((settings.Mint*ucoeff0))*Rd)./real((settings.Mint*ucoeff0));
% e_avg0=ex_avg0;
% D0=Dxx0;
% Vi0=Vix0;

D0_dn_n=Vc*Dzz0.*(real((settings.Mint*ucoeff0))*Rd)./real((settings.Mint*ucoeff0));
e_avg0=ez_avg0;
D0=Dzz0;
Vi0=Viz0;

%% Figure Set up
 f=figure('Position',[20,120,1200,600]); % Control Video Size
 subplot(2,2,1);%subplot(1,2,1);  Left hand Side figure
 a=gca;
 hold on;
 axn1=plot(x,0.5*ones(size(x)),'k-');
 
space_op=diag(e_FK-Vc*Vi_FK)*Rd-Vc*(Rd2*diag(D_FK)+Rd*diag(D_FK*Rd));
cell_den_FK=[zeros(1,N_mesh) 1/dx]/[space_op ones(N_mesh,1)];

% space_op=diag(e_FK)*Rd-Vc*(Rd2*diag(D_FK)+Rd*diag(D_FK*Rd));
% cell_den_GTD=[zeros(1,N_mesh) 1/dx]/[space_op ones(N_mesh,1)];

plot(x,cell_den_FK,'b-.');


 hold off;
 xticks(-1:0.2:1);
%  axis manual;
axis([-1 1 0.4 0.6]);
 
 legend('Exact Smol.','Steady local approx.',...
...%       'location','southoutside','Interpreter','tex','FontSize',16,'Orientation','vertical','NumColumns',2);
     'location','northeast','Interpreter','tex','FontSize',14,'Orientation','vertical','NumColumns',1);
       
%  xlabel('$$x$$','Interpreter','latex','FontSize',16);ylabel('$$n(x)$$','Interpreter','latex','FontSize',16);
 xlabel('$$z$$','Interpreter','latex','FontSize',16,'Position',[0 0.375 -1]);ylabel('$$n(z)$$','Interpreter','latex','FontSize',16);
 a.FontSize=16;a.Position=[0.1008333333333333 0.60425762451325 0.400833333333333 0.377409042153417];%[0.1 0.15 0.392159090909091 0.823333333333333];

 subplot(2,2,[2,4]);%subplot(1,2,2);%%% Right hand Side figure
 a=gca;
%  col=a.ColorOrder;
%  a.ColorOrder=col([1,2,5,3],:);

 hold on;
 axu1=plot(x,e_avg0,'b-','LineWidth',2);
 axu2=plot(x,e_FK,'b--','LineWidth',2);
 axu3=plot(x,Vc*D0,'r-','LineWidth',2);
 axu4=plot(x,Vc*D_FK,'r--','LineWidth',2);
 axu5=plot(x,Vc*Vi0,'g-','LineWidth',2);
 axu6=plot(x,Vc*Vi_FK,'g--','LineWidth',2);
 axu7=plot(x,e_FK,'y-','LineWidth',2);
 axu8=plot(x,D0_dn_n,'k-','LineWidth',1);

 hold off;
 xticks(-1:0.2:1);
 axis manual;

%  legend('$$\langle p_x \rangle_f$$','$$\langle p_x \rangle_g$$',...
%     '$$D_{xx,c}$$','$$Pe_s D_{xx,g,c}$$','$$V_{x,c}$','$$Pe_s V_{x,g,c}$$',...
%     '$$V_{x,\partial t}$$','$$D_{xx,c} (\partial_x n/n)$$',...
%     'Interpreter','latex','location','southoutside','NumColumns',4,'FontSize',16);
 legend('$$\langle p_z \rangle_f$$','$$\langle p_z \rangle_g$$',...
    '$$D_{zz,c}$$','$$Pe_s D_{zz,g,c}$$','$$V_{z,c}$','$$Pe_s V_{z,g,c}$$',...
    '$$V_{z,\partial t}$$','$$D_{zz,c} (\partial_z n/n)$$',...
    'Interpreter','latex','location','southoutside','NumColumns',4,'FontSize',16);
 xlabel('$$z$$','Interpreter','latex','FontSize',16,'Position',[0 -0.15 -1]);%xlabel('$$x$$','Interpreter','latex','FontSize',16,'Position',[0.0,-0.015,-1]); %
%  ylabel('$$\langle p_x \rangle$$ , $$D_{xx}$$ , $$V_x$$ , $$D_{xx} (\partial_x n/n)$$','Interpreter','latex','FontSize',16);
 ylabel('$$\langle p_z \rangle$$ , $$D_{zz}$$ , $$V_z$$ , $$D_{zz} (\partial_z n/n)$$','Interpreter','latex','FontSize',16);
 a.FontSize=16;a.Position=[ 0.579333333333333	0.229333333333333	0.408000000000000	0.695666666666667];%
 title(sprintf('Time t=%2.3f',0.0),'FontSize',18);
 

 subplot(2,2,3);%subplot(1,2,2); % Right hand Side figure
 a3=gca;
%  col=a.ColorOrder;
%  a.ColorOrder=col([1,2,5,3],:);

 hold on;
 axv1=plot(x,zeros(size(x)),'b-','LineWidth',2);
 axv2=plot(x,e_FK,'b--','LineWidth',2);
 axv3=plot(x,Vc*D0,'r-','LineWidth',2);
 axv4=plot(x,Vc*D_FK,'r--','LineWidth',2);
 axv5=plot(x,zeros(size(x)),'g-','LineWidth',2);
 axv6=plot(x,Vc*Vi_FK,'g--','LineWidth',2);
 axv7=plot(x,e_FK,'y-','LineWidth',2);
 axv8=plot(x,zeros(size(x)),'k-','LineWidth',1);

 hold off;
 xticks(-1:0.2:1);
  axis([-1 1 -0.02 0.02]);%axis([-1 1 -0.015 0.035]);
 
 xlabel('$$z$$','Interpreter','latex','FontSize',16,'Position',[0 -0.025 -1]);%xlabel('$$z$$','Interpreter','latex','FontSize',16);
%  ylabel('$$\langle p_x \rangle$$ , $$D_{xx}$$ , $$V_x$$ , $$D_{xx} (\partial_x n/n)$$','Interpreter','latex','FontSize',16);
 ylabel('$$\langle p_z \rangle$$ , $$D_{zz}$$ , $$V_z$$ , $$D_{zz} (\partial_z n/n)$$','Interpreter','latex','FontSize',16);
  a3.FontSize=16;a3.Position=[0.0826    0.0967    0.4215    0.4017];
%% Choosing Data to plot
% cell_den_array=cell_den(1:end/2,:);
% e=Transformed.ex(1:end/2,:);
% D=Transformed.Dxx(1:end/2,:);
% Vi=Transformed.Vix(1:end/2,:);
% Vu=Transformed.Vux(1:end/2,:);
% t_array=t1(1:end/2);

cell_den_array=cell_den(1:end/4,:);
e=Transformed.ez(1:end/4,:);
D=Transformed.Dzz(1:end/4,:);
Vi=Transformed.Viz(1:end/4,:);
Vu=Transformed.Vuz(1:end/4,:);
t_array=t1(:,1:end/4)';

%% Script to generate video
v = VideoWriter(['beta'  num2str(beta) '_B' num2str(B) 'HS.avi'],'Motion JPEG AVI');
v.FrameRate = 30; % Speed control, higher means faster

open(v);
% 1st Frame (t=0)
 for i=1:60  
 drawnow;
 F=getframe(f);
 writeVideo(v,F);
 end
% Subsequest Frames
%   subplot(2,2,[2,4]); % subplot(1,2,2);
 for i=1:length(t_array) % More steps between frame means faster and less resolution in time.
     axn1.YData=cell_den_array(i,:); %Fastest way to update a figure
%      axn2.YData=W_x(i,:);
     
     axu1.YData=real(e(i,:));
     axu3.YData=Vc*real(D(i,:));
     axu5.YData=Vc*real(Vi(i,:));
     axu7.YData=real(Vu(i,:));
     axu8.YData=Vc*real(D(i,:).*(cell_den_array(i,:)*Rd)./cell_den_array(i,:));
     title(a,sprintf('Time t=%2.3f',t_array(i)),'FontSize',18);
     
     axv1.YData=real(e(i,:));
     axv3.YData=Vc*real(D(i,:));
     axv5.YData=Vc*real(Vi(i,:));
     axv7.YData=real(Vu(i,:));
     axv8.YData=Vc*real(D(i,:).*(cell_den_array(i,:)*Rd)./cell_den_array(i,:));
     
     drawnow;
     F=getframe(f);
     writeVideo(v,F);
%      F=getframe(f);
%      writeVideo(v,F);
%      F=getframe(f);
%      writeVideo(v,F);
%      F=getframe(f);
%      writeVideo(v,F);
%      F=getframe(f);
%      writeVideo(v,F);
%      F=getframe(f);
%      writeVideo(v,F);
 end
%  for i=21:70 % More steps between frame means faster and less resolution in time.
%      axn1.YData=cell_den_array(i,:); %Fastest way to update a figure
% %      axn2.YData=W_x(i,:);
%      
%      axu1.YData=real(ex(i,:));
%      axu3.YData=Vc*real(Dxx(i,:));
%      axu5.YData=Vc*real(Vix(i,:));
%      axu7.YData=real(Vux(i,:));
%      axu8.YData=Vc*real(Dxx(i,:).*(cell_den_array(i,:)*Rd)./cell_den_array(i,:));
%      title(sprintf('Time t=%2.3f',t_array(i)),'FontSize',18);
%      
% %      axv1.YData=real(ex(i,:));
% %      axv3.YData=Vc*real(Dxx(i,:));
% %      axv5.YData=Vc*real(Vix(i,:));
% %      axv7.YData=real(Vux(i,:));
% %      axv8.YData=Vc*real(Dxx(i,:).*(cell_den_array(i,:)*Rd)./cell_den_array(i,:));
%      
%      drawnow;
%      F=getframe(f);
%      writeVideo(v,F);
% %      F=getframe(f);
% %      writeVideo(v,F);
% %      F=getframe(f);
% %      writeVideo(v,F);
%  end 
% for i=71:length(t_array) % More steps between frame means faster and less resolution in time.
%      axn1.YData=cell_den_array(i,:); %Fastest way to update a figure
% %      axn2.YData=W_x(i,:);
%      
%      axu1.YData=real(ex(i,:));
%      axu3.YData=Vc*real(Dxx(i,:));
%      axu5.YData=Vc*real(Vix(i,:));
%      axu7.YData=real(Vux(i,:));
%      axu8.YData=Vc*real(Dxx(i,:).*(cell_den_array(i,:)*Rd)./cell_den_array(i,:));
%      title(sprintf('Time t=%2.3f',t_array(i)),'FontSize',18);
%      
% %      axv1.YData=real(ex(i,:));
% %      axv3.YData=Vc*real(Dxx(i,:));
% %      axv5.YData=Vc*real(Vix(i,:));
% %      axv7.YData=real(Vux(i,:));
% %      axv8.YData=Vc*real(Dxx(i,:).*(cell_den_array(i,:)*Rd)./cell_den_array(i,:));
%      
%      drawnow;
%      F=getframe(f);
%      writeVideo(v,F);
% end 

% Save the video
 close(v);