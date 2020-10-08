% On-the-go-Post-Processing for transformed variables
function  [ex_avg,ez_avg,Dxx_temp,Dzx_temp,Dxz_temp,Dzz_temp,Vix_temp,Viz_temp,...
    VDTx_temp,VDTz_temp,DDTxx_temp,DDTzx_temp]=PS_transformed(f,CPUPS)
        d2xf=f*CPUPS.Rd2x;
        dxf=f*CPUPS.Rdx;
%         d2zf=f*CPUPS.Rd2z;
%         dzf=f*CPUPS.Rdz;
        ex_avg=real(CPUPS.Mint*CPUPS.Mp1*f*(2*pi));
        ez_avg=real(CPUPS.Mint*CPUPS.Mp3*f*(2*pi));
%         exz_avg=real(CPUPS.Mint*CPUPS.Mp1p3*f*(2*pi));
%         ezz_avg=real(CPUPS.Mint*CPUPS.Mp3sq*f*(2*pi));
        
        bx_RHS=CPUPS.Mp1*f-ex_avg.*f;
        bz_RHS=CPUPS.Mp3*f-ez_avg.*f;
%         inhomo_RHS=CPUPS.Mp1*(dxf)+CPUPS.Mp3*(dzf)-(ex_avg*CPUPS.Rdx+ez_avg*CPUPS.Rdz).*f;
        inhomo_RHS=CPUPS.Mp1*(dxf)-(ex_avg*CPUPS.Rdx).*f;
%         b_swimvar_x_RHS=CPUPS.Mp1p3*f-exz_avg.*f;
%         b_swimvar_z_RHS=CPUPS.Mp3sq*f-ezz_avg.*f;
%         f_swimvari_RHS=CPUPS.Mp1p3*dxf+CPUPS.Mp3sq*dzf-(exz_avg*CPUPS.Rdx+ezz_avg*CPUPS.Rdz).*f;
        
        Dxx_temp    =NaN(1,CPUPS.Nx_mesh);
        Dzx_temp    =NaN(1,CPUPS.Nx_mesh);
        Dxz_temp    =NaN(1,CPUPS.Nx_mesh);
        Dzz_temp    =NaN(1,CPUPS.Nx_mesh);
        Vix_temp    =NaN(1,CPUPS.Nx_mesh);
        Viz_temp    =NaN(1,CPUPS.Nx_mesh);
        VDTx_temp   =NaN(1,CPUPS.Nx_mesh);
        VDTz_temp   =NaN(1,CPUPS.Nx_mesh);
%         Vax_temp    =NaN(1,CPUPS.Nx_mesh);
%         Vaz_temp    =NaN(1,CPUPS.Nx_mesh);
        DDTxx_temp  =NaN(1,CPUPS.Nx_mesh);
        DDTzx_temp  =NaN(1,CPUPS.Nx_mesh);
%         DDTxz_temp  =NaN(1,CPUPS.Nx_mesh);
%         DDTzz_temp  =NaN(1,CPUPS.Nx_mesh);
%         Vswimminx_temp=NaN(1,CPUPS.Nx_mesh);
%         Vswimminz_temp=NaN(1,CPUPS.Nx_mesh);
%         Dswimxx_temp=NaN(1,CPUPS.Nx_mesh);
%         Dswimxz_temp=NaN(1,CPUPS.Nx_mesh);
%         Dswimzx_temp=NaN(1,CPUPS.Nx_mesh);
%         Dswimzz_temp=NaN(1,CPUPS.Nx_mesh);
%         Vswimvarx_temp=NaN(1,CPUPS.Nx_mesh);
%         Vswimvarz_temp=NaN(1,CPUPS.Nx_mesh);
        
        
        for j=1:CPUPS.Nx_mesh           

                     bx=time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile(j),...
                         bx_RHS(:,j),CPUPS.Mint,CPUPS.MintSq,CPUPS.n,CPUPS.m,CPUPS.dt);
                     bz=time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile(j),...
                         bz_RHS(:,j),CPUPS.Mint,CPUPS.MintSq,CPUPS.n,CPUPS.m,CPUPS.dt);
                b_DT_p1=time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile(j),...
                         dxf(:,j),CPUPS.Mint,CPUPS.MintSq,CPUPS.n,CPUPS.m,CPUPS.dt);
%                 b_DT_p3=time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile(j),...
%                          dzf(:,j),CPUPS.Mint,CPUPS.MintSq,CPUPS.n,CPUPS.m,CPUPS.dt);                     
               f_inhomo=time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile(j),...
                         inhomo_RHS(:,j),CPUPS.Mint,CPUPS.MintSq,CPUPS.n,CPUPS.m,CPUPS.dt);
                   f_DT=time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile(j),...
                         d2xf(:,j),CPUPS.Mint,CPUPS.MintSq,CPUPS.n,CPUPS.m,CPUPS.dt);
%                    f_DT=dLe\(d2xf(:,j)+d2zf(:,j));
%                     f_a=dLe\(CPUPS.U_profile(j)*dxf(:,j)+CPUPS.W_profile(j)*dzf(:,j));
%               f_swimmin=dLe\[dzf(:,j);0];
%             b_swimvar_x=dLe\b_swimvar_x_RHS(:,j);
%             b_swimvar_z=dLe\b_swimvar_z_RHS(:,j);
%             f_swimvar_i=dLe\f_swimvari_RHS(:,j) ;

             Dxx_temp(j)=CPUPS.Mint*(CPUPS.Mp1*bx)*2*pi;
             Dxz_temp(j)=CPUPS.Mint*(CPUPS.Mp1*bz)*2*pi;
             Dzx_temp(j)=CPUPS.Mint*(CPUPS.Mp3*bx)*2*pi;
             Dzz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*bz)*2*pi;
             Vix_temp(j)=CPUPS.Mint*(CPUPS.Mp1*f_inhomo)*2*pi;
             Viz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*f_inhomo)*2*pi;            
            VDTx_temp(j)=CPUPS.Mint*(CPUPS.Mp1*f_DT)*2*pi;
            VDTz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*f_DT)*2*pi;
%              Vax_temp(j)=CPUPS.Mint*(CPUPS.Mp1*f_a)*2*pi;
%              Vaz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*f_a)*2*pi;
             
%             Vswimminx_temp(j)=CPUPS.Mint*(CPUPS.Mp1*f_swimmin)*2*pi;
%             Vswimminz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*f_swimmin)*2*pi;
%             Dswimxx_temp(j)=CPUPS.Mint*(CPUPS.Mp1*b_swimvar_x)*2*pi;
%             Dswimxz_temp(j)=CPUPS.Mint*(CPUPS.Mp1*b_swimvar_z)*2*pi;
%             Dswimzx_temp(j)=CPUPS.Mint*(CPUPS.Mp3*b_swimvar_x)*2*pi;
%             Dswimzz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*b_swimvar_z)*2*pi;
%             Vswimvarx_temp(j)=CPUPS.Mint*(CPUPS.Mp1*f_swimvar_i)*2*pi;
%             Vswimvarz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*f_swimvar_i)*2*pi;
            
            DDTxx_temp(j)=CPUPS.Mint*(CPUPS.Mp1*b_DT_p1)*2*pi;
            DDTzx_temp(j)=CPUPS.Mint*(CPUPS.Mp3*b_DT_p1)*2*pi;
%             DDTxz_temp(j)=CPUPS.Mint*(CPUPS.Mp1*b_DT_p3)*2*pi;
%             DDTzz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*b_DT_p3)*2*pi;
            
        end

end

function ucoeff=time_relaxed_Linv_f(Mvor,Mgyro,Mlap,S,forcing,Mint,MintSq,n,m,dt)

ucoeff=zeros(n*m,1);
    
Madv=S*(Mvor)+Mgyro;

%% Loop!
ucoeffp=zeros(n*m,1);
% adv_coeff=Madv*ucoeff;
% lap_coeff=Mlap*ucoeff;
i=1;
while (norm((ucoeff-ucoeffp)/dt)>1e-8 || i<20) && i<5000
    ucoeffp=ucoeff;
    adv_coeff=Madv*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;


    rhs_coeff = -2/dt*ucoeff-lap_coeff+2*(adv_coeff-forcing);
    ucoeff = helmholtz_coeff(rhs_coeff, -2/dt, n, m);
    i=i+1;

end
% disp([num2str(i) '    ' num2str(norm((ucoeff-ucoeffp)/dt)) '    ' num2str(abs(Mint*ucoeff*2*pi))]);
end