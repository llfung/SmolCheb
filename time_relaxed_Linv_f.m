% On-the-go-Post-Processing for transformed variables
function  [ex,ez,Dxx,Dxz,Dzx,Dzz,Vix,Viz,...
    VDTx,VDTz,DDTx,DDTz,Vdeltx,Vdeltz]=time_relaxed_Linv_f(dir,settings,S_profile,f,fdt)

helm=helmholtz_genGPU( settings.n, settings.m);
helm.dt=settings.dt;
[settings,Mvor,Mgyro,Mlap,Rd,Rd2,Mp1,Mp3]=all_mat_gen(settings);

Mint=settings.Mint;
MintSq=settings.MintSq;

d2f=f*Rd2;
df=f*Rd;

ex=real(Mint*Mp1*f*(2*pi));
ez=real(Mint*Mp3*f*(2*pi));
% exz_avg=real(Mint*Mp1p3*f*(2*pi));
% ezz_avg=real(Mint*Mp3sq*f*(2*pi));

bx_RHS=Mp1*f-ex.*f;
bz_RHS=Mp3*f-ez.*f;

switch dir
    case 'x'
        inhomo_RHS=Mp1*(df)-(ex*Rd).*f;
    case 'z'
        inhomo_RHS=Mp3*(df)-(ez*Rd).*f;
    otherwise
        error('Linv_f: unknown direction');
end
% b_swimvar_x_RHS=Mp1p3*f-exz_avg.*f;
% b_swimvar_z_RHS=Mp3sq*f-ezz_avg.*f;
% f_swimvari_RHS=Mp1p3*dxf+Mp3sq*dzf-(exz_avg*Rdx+ezz_avg*Rdz).*f;

bx = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
    bx_RHS,Mint,MintSq,Mp1,Mp3,helm);
bz = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
    bz_RHS,Mint,MintSq,Mp1,Mp3,helm);
b_DT = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
    df,Mint,MintSq,Mp1,Mp3,helm);
f_inhomo = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
    inhomo_RHS,Mint,MintSq,Mp1,Mp3,helm);
f_DT = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
    d2f,Mint,MintSq,Mp1,Mp3,helm);
% f_u = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
%     U_profile(j).*dxf+W_profile.*dzf,Mint,MintSq,Mp1,Mp3,Nx_mesh,helm);

Dxx=real(Mint*(Mp1*bx)*2*pi);
Dxz=real(Mint*(Mp1*bz)*2*pi);
Dzx=real(Mint*(Mp3*bx)*2*pi);
Dzz=real(Mint*(Mp3*bz)*2*pi);
Vix=real(Mint*(Mp1*f_inhomo)*2*pi);
Viz=real(Mint*(Mp3*f_inhomo)*2*pi);
VDTx=real(Mint*(Mp1*f_DT)*2*pi);
VDTz=real(Mint*(Mp3*f_DT)*2*pi);
DDTx=real(Mint*(Mp1*b_DT)*2*pi);
DDTz=real(Mint*(Mp3*b_DT)*2*pi);
% Vux=real(Mint*(Mp1*f_u)*2*pi);
% Vuz=real(Mint*(Mp3*f_u)*2*pi);

% Vswimminx=real(Mint*(Mp1*f_swimmin)*2*pi);
% Vswimminz=real(Mint*(Mp3*f_swimmin)*2*pi);
% Dswimxx=real(Mint*(Mp1*b_swimvar_x)*2*pi);
% Dswimxz=real(Mint*(Mp1*b_swimvar_z)*2*pi);
% Dswimzx=real(Mint*(Mp3*b_swimvar_x)*2*pi);
% Dswimzz=real(Mint*(Mp3*b_swimvar_z)*2*pi);
% Vswimvarx=real(Mint*(Mp1*f_swimvar_i)*2*pi);
% Vswimvarz=real(Mint*(Mp3*f_swimvar_i)*2*pi);

if nargin>3
    f_delt = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
        fdt,Mint,MintSq,Mp1,Mp3,helm);
    Vdeltx = real(Mint*(Mp1*f_delt))*2*pi;
    Vdeltz = real(Mint*(Mp3*f_delt))*2*pi;
end
end


