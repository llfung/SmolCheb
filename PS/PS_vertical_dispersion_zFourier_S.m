S=[0:0.25:2.75];
for jj=1:length(S)
    a=dir(['D:\db\Smol\2D_VS\smol_pBC_2D_*epsInit_0beta_0B_0Vsm_0Vsv_0.0*Vc_0DT_' num2str(S(jj)) 'Pef_homo_dx*']);
    load([a(1).folder '\' a(1).name]);
    
epsInitx=epsInit;
epsInitz=epsInit;
rhoxz=0;
sigmax=sqrt(2*epsInitx);
sigmaz=sqrt(2*epsInitz);

cell_phy=real(ifft(reshape(cell_den,length(t3),Nx_mesh,Nz_mesh),[],3))*Nz_mesh;

varxx=zeros(1,length(t3));
varzz=zeros(1,length(t3));
varxz=zeros(1,length(t3));
varzz_x=zeros(1,length(t3));
cell_den_loc=NaN(Nz_mesh+1,Nx_mesh+1);
M1zmap=zeros(1,length(t3));
M2zmap=zeros(1,length(t3));
varzzmapt=zeros(1,length(t3));
Uxt=zeros(1,length(t3));
Uzt=zeros(1,length(t3));
M2zt=zeros(1,length(t3));

M1z_x=NaN(length(t3),Nx_mesh+1);
xint=[x x_width/2];
zint=[z';z_width/2];
for i=1:length(t3)
    zmap=zint-Pef*xint*t3(i);
%     cell_den_loc(1:Nz_mesh,1:Nx_mesh)=reshape(cell_den(i,:),Nz_mesh,Nx_mesh);
    cell_den_loc(1:Nz_mesh,1:Nx_mesh)=real(transpose(reshape(cell_phy(i,:,:),Nx_mesh,Nz_mesh)));
    cell_den_loc(Nz_mesh+1,:)=cell_den_loc(1,:);
    cell_den_loc(:,Nx_mesh+1)=cell_den_loc(:,1);
    M1x=trapz(zint,trapz(xint,cell_den_loc.*xint,2),1);
    M2x=trapz(zint,trapz(xint,cell_den_loc.*(xint.^2),2),1);
    M1z=trapz(xint,trapz(zint,cell_den_loc.*zint,1),2);
    M1zmap(i)=trapz(xint,trapz(zint,cell_den_loc.*zmap,1),2);
    M1z_x=trapz(zint,cell_den_loc.*zint,1);
    M2z=trapz(xint,trapz(zint,cell_den_loc.*(zint.^2),1),2);
    M2zmap(i)=trapz(xint,trapz(zint,cell_den_loc.*(zmap.^2),1),2);
    Mxz=trapz(xint,trapz(zint,cell_den_loc.*zint,1).*xint,2);
    varzz_x(i)=trapz(zint,trapz(xint,cell_den_loc.*(zint-M1z_x).^2,2),1);
    varzzmapt(i)=trapz(zint,trapz(xint,cell_den_loc.*(zmap-M1zmap(i)).^2,2),1);
    Uxt(i)=M1x;
    Uzt(i)=M1z;
    M2zt(i)=M2z;
    varxx(i)=M2x-M1x*M1x;
    varzz(i)=M2z-M1z*M1z;
    varxz(i)=Mxz-M1x*M1z;
end

twoDxzt=varxz-(varxx/2+sigmax^2/2).*t3*Pef;
twoDzzt=varzz-(varxx+2*sigmax^2).*((Pef*t3).^2/3)-Pef*t3.*(twoDxzt-rhoxz*sigmax*sigmaz)-2*Pef*rhoxz*sigmax*sigmaz*t3;

varxx_dt=(varxx(3:end)-varxx(1:end-2))./(t3(3:end)-t3(1:end-2));
varxz_dt=(varxz(3:end)-varxz(1:end-2))./(t3(3:end)-t3(1:end-2));
varzz_dt=(varzz(3:end)-varzz(1:end-2))./(t3(3:end)-t3(1:end-2));
twoDxz=(twoDxzt(3:end)-twoDxzt(1:end-2))./(t3(3:end)-t3(1:end-2));
twoDzz=(twoDzzt(3:end)-twoDzzt(1:end-2))./(t3(3:end)-t3(1:end-2));
varzz_x_dt=(varzz_x(3:end)-varzz_x(1:end-2))./(t3(3:end)-t3(1:end-2));

dt2=(mean(t3(3:end)-t3(1:end-2))/2)^2;
varxx_d2t=(varxx(3:end)-2*varxx(2:end-1)+varxx(1:end-2))./dt2;
varxz_d2t=(varxz(3:end)-2*varxz(2:end-1)+varxz(1:end-2))./dt2;
varzz_d2t=(varzz(3:end)-2*varzz(2:end-1)+varzz(1:end-2))./dt2;
twoDxz_dt=(twoDxzt(3:end)-2*twoDxzt(2:end-1)+twoDxzt(1:end-2))./dt2;
twoDzz_dt=(twoDzzt(3:end)-2*twoDzzt(2:end-1)+twoDzzt(1:end-2))./dt2;

Dxx_oldroyd=varxx_dt/2;
Dxz_oldroyd=(varxz_dt-Pef*varxx(2:end-1))/2;
Dzz_oldroyd=(varzz_dt-2*Pef*varxz(2:end-1))/2;

Dxx_oldroyd_S(jj)=Dxx_oldroyd(end/2)/Vc/Vc;
Dxz_oldroyd_S(jj)=Dxz_oldroyd(end/2)/Vc/Vc;
Dzz_oldroyd_S(jj)=Dzz_oldroyd(end/2)/Vc/Vc;
end