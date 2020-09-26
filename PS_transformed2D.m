% On-the-go-Post-Processing for transformed variables
function [ex_avg,ez_avg,exz_avg,ezz_avg,Dxx_temp,Dxz_temp,Dzx_temp,Dzz_temp,Vix_temp,Viz_temp,...
    VDTx_temp,VDTz_temp,DDTxx_temp,DDTxz_temp,DDTzx_temp,DDTzz_temp,...
    Vax_temp,Vaz_temp,Vswimvarx_temp,Vswimvarz_temp,...
    Dswimxx_temp,Dswimxz_temp,Dswimzx_temp,Dswimzz_temp]=PS_transformed2D(f,CPUPS)
        d2xf=f*CPUPS.Rd2x;
        dxf=f*CPUPS.Rdx;
        d2zf=f*CPUPS.Rd2z;
        dzf=f*CPUPS.Rdz;
        ex_avg=real(CPUPS.Mint*CPUPS.Mp1*f*(2*pi));
        ez_avg=real(CPUPS.Mint*CPUPS.Mp3*f*(2*pi));
        exz_avg=real(CPUPS.Mint*CPUPS.Mp1p3*f*(2*pi));
        ezz_avg=real(CPUPS.Mint*CPUPS.Mp3sq*f*(2*pi));
        
        bx_RHS=CPUPS.Mp1*f-ex_avg.*f;
        bz_RHS=CPUPS.Mp3*f-ez_avg.*f;
        inhomo_RHS=CPUPS.Mp1*(dxf)+CPUPS.Mp3*(dzf)-(ex_avg*CPUPS.Rdx+ez_avg*CPUPS.Rdz).*f;
        b_swimvar_x_RHS=CPUPS.Mp1p3*f-exz_avg.*f;
        b_swimvar_z_RHS=CPUPS.Mp3sq*f-ezz_avg.*f;
        f_swimvari_RHS=CPUPS.Mp1p3*dxf+CPUPS.Mp3sq*dzf-(exz_avg*CPUPS.Rdx+ezz_avg*CPUPS.Rdz).*f;
        
        Dxx_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Dzx_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Dxz_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Dzz_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Vix_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Viz_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        VDTx_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        VDTz_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Vax_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Vaz_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        DDTxx_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        DDTxz_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        DDTzx_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        DDTzz_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
%         Vswimminx_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
%         Vswimminz_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Dswimxx_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Dswimxz_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Dswimzx_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Dswimzz_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Vswimvarx_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        Vswimvarz_temp=NaN(1,CPUPS.Nz_mesh*CPUPS.Nx_mesh);
        
        zeroth_row=CPUPS.m*CPUPS.n/2+CPUPS.m/2+1;
        bx_RHS=CPUPS.Msin2*bx_RHS;
        bx_RHS(zeroth_row,:)=0;
        bz_RHS=CPUPS.Msin2*bz_RHS;
        bz_RHS(zeroth_row,:)=0;
        dxf=CPUPS.Msin2*dxf;
        dxf(zeroth_row,:)=0;
        dzf=CPUPS.Msin2*dzf;
        dzf(zeroth_row,:)=0;
        d2xf=CPUPS.Msin2*d2xf;
        d2xf(zeroth_row,:)=0;
        d2zf=CPUPS.Msin2*d2zf;
        d2zf(zeroth_row,:)=0;   
        inhomo_RHS=CPUPS.Msin2*inhomo_RHS;
        inhomo_RHS(zeroth_row,:)=0;   
        b_swimvar_x_RHS=CPUPS.Msin2*b_swimvar_x_RHS;
        b_swimvar_x_RHS(zeroth_row,:)=0;   
        b_swimvar_z_RHS=CPUPS.Msin2*b_swimvar_z_RHS;
        b_swimvar_z_RHS(zeroth_row,:)=0;  
        f_swimvari_RHS=CPUPS.Msin2*f_swimvari_RHS;
        f_swimvari_RHS(zeroth_row,:)=0;  
        
        for j=1:CPUPS.Nz_mesh*CPUPS.Nx_mesh           
            Le=CPUPS.omg2_profile(j)*CPUPS.Mvor+CPUPS.e11_profile(j)*CPUPS.Me11...
                +CPUPS.e13_profile(j)*CPUPS.Me13+CPUPS.e33_profile(j)*CPUPS.Me33...
                +CPUPS.Mgyro-CPUPS.Mlap;
            Le=CPUPS.Msin2*Le;
            Le(zeroth_row,:)=CPUPS.Mint;
            dLe=decomposition(Le);
%                dLe=dLe_array{j};
                     bx=dLe\bx_RHS(:,j);
                     bz=dLe\bz_RHS(:,j);
                b_DT_p1=dLe\dxf(:,j);
                b_DT_p3=dLe\dzf(:,j);
               f_inhomo=dLe\inhomo_RHS(:,j);
                   f_DT=dLe\(d2xf(:,j)+d2zf(:,j));
                    f_a=dLe\(CPUPS.U_profile(j)*dxf(:,j)+CPUPS.W_profile(j)*dzf(:,j));
%               f_swimmin=dLe\[dzf(:,j);0];
            b_swimvar_x=dLe\b_swimvar_x_RHS(:,j);
            b_swimvar_z=dLe\b_swimvar_z_RHS(:,j);
            f_swimvar_i=dLe\f_swimvari_RHS(:,j) ;

             Dxx_temp(j)=CPUPS.Mint*(CPUPS.Mp1*bx)*2*pi;
             Dxz_temp(j)=CPUPS.Mint*(CPUPS.Mp1*bz)*2*pi;
             Dzx_temp(j)=CPUPS.Mint*(CPUPS.Mp3*bx)*2*pi;
             Dzz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*bz)*2*pi;
             Vix_temp(j)=CPUPS.Mint*(CPUPS.Mp1*f_inhomo)*2*pi;
             Viz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*f_inhomo)*2*pi;            
            VDTx_temp(j)=CPUPS.Mint*(CPUPS.Mp1*f_DT)*2*pi;
            VDTz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*f_DT)*2*pi;
             Vax_temp(j)=CPUPS.Mint*(CPUPS.Mp1*f_a)*2*pi;
             Vaz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*f_a)*2*pi;
             
%             Vswimminx_temp(j)=CPUPS.Mint*(CPUPS.Mp1*f_swimmin)*2*pi;
%             Vswimminz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*f_swimmin)*2*pi;
            Dswimxx_temp(j)=CPUPS.Mint*(CPUPS.Mp1*b_swimvar_x)*2*pi;
            Dswimxz_temp(j)=CPUPS.Mint*(CPUPS.Mp1*b_swimvar_z)*2*pi;
            Dswimzx_temp(j)=CPUPS.Mint*(CPUPS.Mp3*b_swimvar_x)*2*pi;
            Dswimzz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*b_swimvar_z)*2*pi;
            Vswimvarx_temp(j)=CPUPS.Mint*(CPUPS.Mp1*f_swimvar_i)*2*pi;
            Vswimvarz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*f_swimvar_i)*2*pi;
            
            DDTxx_temp(j)=CPUPS.Mint*(CPUPS.Mp1*b_DT_p1)*2*pi;
            DDTxz_temp(j)=CPUPS.Mint*(CPUPS.Mp1*b_DT_p3)*2*pi;
            DDTzx_temp(j)=CPUPS.Mint*(CPUPS.Mp3*b_DT_p1)*2*pi;
            DDTzz_temp(j)=CPUPS.Mint*(CPUPS.Mp3*b_DT_p3)*2*pi;
            
        end

end