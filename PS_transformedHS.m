% On-the-go-Post-Processing for transformed variables
function  [ex_avg,ez_avg,Dxx_temp,Dzx_temp,Dxz_temp,Dzz_temp,Vix_temp,Viz_temp,...
    VDTx_temp,VDTz_temp,DDTxz_temp,DDTzz_temp]=PS_transformedHS(f,CPUPS,helm)
        d2zf=f*CPUPS.Rd2z;
        dzf=f*CPUPS.Rdz;
%         d2zf=f*CPUPS.Rd2z;
%         dzf=f*CPUPS.Rdz;
        ex_avg=real(CPUPS.Mint*CPUPS.Mp1*f*(2*pi));
        ez_avg=real(CPUPS.Mint*CPUPS.Mp3*f*(2*pi));
%         exz_avg=real(CPUPS.Mint*CPUPS.Mp1p3*f*(2*pi));
%         ezz_avg=real(CPUPS.Mint*CPUPS.Mp3sq*f*(2*pi));
        
        bx_RHS=CPUPS.Mp1*f-ex_avg.*f;
        bz_RHS=CPUPS.Mp3*f-ez_avg.*f;
%         inhomo_RHS=CPUPS.Mp1*(dxf)+CPUPS.Mp3*(dzf)-(ex_avg*CPUPS.Rdx+ez_avg*CPUPS.Rdz).*f;
        inhomo_RHS=CPUPS.Mp3*(dzf)-(ez_avg*CPUPS.Rdz).*f;
%         b_swimvar_x_RHS=CPUPS.Mp1p3*f-exz_avg.*f;
%         b_swimvar_z_RHS=CPUPS.Mp3sq*f-ezz_avg.*f;
%         f_swimvari_RHS=CPUPS.Mp1p3*dxf+CPUPS.Mp3sq*dzf-(exz_avg*CPUPS.Rdx+ezz_avg*CPUPS.Rdz).*f;

                 bx = time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile,...
                     bx_RHS,CPUPS.Mint,CPUPS.MintSq,CPUPS.Mp1,CPUPS.Mp3,CPUPS.Nz_mesh,helm);
                 bz = time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile,...
                     bz_RHS,CPUPS.Mint,CPUPS.MintSq,CPUPS.Mp1,CPUPS.Mp3,CPUPS.Nz_mesh,helm);
%             b_DT_p1 = time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile,...
%                      dxf,CPUPS.Mint,CPUPS.MintSq,CPUPS.Mp1,CPUPS.Mp3,CPUPS.Nz_mesh,helm);
            b_DT_p3 = time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile,...
                         dzf,CPUPS.Mint,CPUPS.MintSq,CPUPS.Mp1,CPUPS.Mp3,CPUPS.Nz_mesh,helm);                  
           f_inhomo = time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile,...
                     inhomo_RHS,CPUPS.Mint,CPUPS.MintSq,CPUPS.Mp1,CPUPS.Mp3,CPUPS.Nz_mesh,helm);
               f_DT = time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile,...
                     d2zf,CPUPS.Mint,CPUPS.MintSq,CPUPS.Mp1,CPUPS.Mp3,CPUPS.Nz_mesh,helm);
%                 f_a = time_relaxed_Linv_f(CPUPS.Mvor,CPUPS.Mgyro,CPUPS.Mlap,CPUPS.S_profile,...
%                          CPUPS.U_profile.*dxf+CPUPS.W_profile.*dzf,CPUPS.Mint,CPUPS.MintSq,CPUPS.Mp1,CPUPS.Mp3,CPUPS.Nz_mesh,helm);  

             Dxx_temp=CPUPS.Mint*(CPUPS.Mp1*bx)*2*pi;
             Dxz_temp=CPUPS.Mint*(CPUPS.Mp1*bz)*2*pi;
             Dzx_temp=CPUPS.Mint*(CPUPS.Mp3*bx)*2*pi;
             Dzz_temp=CPUPS.Mint*(CPUPS.Mp3*bz)*2*pi;
             Vix_temp=CPUPS.Mint*(CPUPS.Mp1*f_inhomo)*2*pi;
             Viz_temp=CPUPS.Mint*(CPUPS.Mp3*f_inhomo)*2*pi;            
            VDTx_temp=CPUPS.Mint*(CPUPS.Mp1*f_DT)*2*pi;
            VDTz_temp=CPUPS.Mint*(CPUPS.Mp3*f_DT)*2*pi;
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
            
%             DDTxx_temp=CPUPS.Mint*(CPUPS.Mp1*b_DT_p1)*2*pi;
%             DDTzx_temp=CPUPS.Mint*(CPUPS.Mp3*b_DT_p1)*2*pi;
            DDTxz_temp=CPUPS.Mint*(CPUPS.Mp1*b_DT_p3)*2*pi;
            DDTzz_temp=CPUPS.Mint*(CPUPS.Mp3*b_DT_p3)*2*pi;
            
        

end

function ucoeff=time_relaxed_Linv_f(Mvor,Mgyro,Mlap,S,forcing,Mint,MintSq,Mp1,Mp3,N_mesh,helm)
%% Initialisation
ucoeff=zeros(size(forcing));

%% Parameters
forcing=forcing-Mint'*(Mint*forcing)/MintSq;

%% RK3 coeff and constants
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];
K2=1/helm.dt;
%% Loop!
epsilon=5e-5;

for i=1:10000
    
    ucoeffp=ucoeff;
    
    k=1;
    adv_coeff=S.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;

    adv_p_coeff=adv_coeff-forcing;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/alpha(k)*gamma(k))*adv_p_coeff;
    
    F = permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj  = (-alpha(k)/K2)*sum(helm.en'.*F(:,helm.k,:),1);
    Fhorizontal = reshape(F,helm.m,helm.n*N_mesh);
    Fhorizontal = (-alpha(k)/K2)*helm.L2*Fhorizontal;
    F = reshape(Fhorizontal,helm.m,helm.n,N_mesh);
    F(helm.floorm+1,helm.k,:) = int_constj;
    
    CFS = helm.helm_inv_k1*reshape(F,helm.n*helm.m,N_mesh);
    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);  
    
    k=2;
    adv_coeff=S.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;

    adv_comb_coeff=adv_coeff-forcing;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/alpha(k)*gamma(k))*adv_comb_coeff+(1/alpha(k)*rho(k))*adv_p_coeff;

    F = permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj  = (-alpha(k)/K2)*sum(helm.en'.*F(:,helm.k,:),1);
    Fhorizontal = reshape(F,helm.m,helm.n*N_mesh);
    Fhorizontal = (-alpha(k)/K2)*helm.L2*Fhorizontal;
    F = reshape(Fhorizontal,helm.m,helm.n,N_mesh);
    F(helm.floorm+1,helm.k,:) = int_constj;
    
    CFS = helm.helm_inv_k2*reshape(F,helm.n*helm.m,N_mesh);
    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);  
    
    k=3;
    adv_p_coeff=adv_comb_coeff;
    adv_coeff=S.*(Mvor*ucoeff)+Mgyro*ucoeff;
    adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
    
    lap_coeff=Mlap*ucoeff;
    lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;

    adv_comb_coeff=adv_coeff-forcing;
    rhs_coeff = (-K2/alpha(k))*ucoeff-lap_coeff+(1/alpha(k)*gamma(k))*adv_comb_coeff+(1/alpha(k)*rho(k))*adv_p_coeff;

    F = permute(reshape(rhs_coeff,helm.n,helm.m,N_mesh),[2 1 3]);
    int_constj  = (-alpha(k)/K2)*sum(helm.en'.*F(:,helm.k,:),1);
    Fhorizontal = reshape(F,helm.m,helm.n*N_mesh);
    Fhorizontal = (-alpha(k)/K2)*helm.L2*Fhorizontal;
    F = reshape(Fhorizontal,helm.m,helm.n,N_mesh);
    F(helm.floorm+1,helm.k,:) = int_constj;

    CFS = helm.helm_inv_k3*reshape(F,helm.n*helm.m,N_mesh);
    ucoeff=reshape(permute(reshape(CFS,helm.m,helm.n,N_mesh),[2 1 3]),helm.n*helm.m,N_mesh);  
    
    if ~mod(i,25)
        %TODO: better calculation of norm
        errMp1=max(abs(Mint*(Mp1*(ucoeff-ucoeffp))))*2*pi/helm.dt;
        errMp3=max(abs(Mint*(Mp3*(ucoeff-ucoeffp))))*2*pi/helm.dt;
        if max(errMp1,errMp3)<epsilon
            break;
        end
    end
end

% disp([num2str(i) '    ' num2str(errMp1) '  ' num2str(errMp3) '    ' num2str(max(abs(Mint*ucoeff))*2*pi)]);
end