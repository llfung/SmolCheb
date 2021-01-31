% On-the-go-Post-Processing for transformed variables
function  [ex,ez,Dxx,Dxz,Dzx,Dzz,Vix,Viz,...
    VDTx,VDTz,DDTx,DDTz,Vdeltx,Vdeltz]=time_relaxed_Linv_f(dir,settings,S_profile,f,fdt)
    persistent bx bz b_DT f_inhomo f_DT f_delt
if isempty(bx)
    bx=zeros(settings.n*settings.m,settings.N_mesh);
    bz=zeros(settings.n*settings.m,settings.N_mesh);
    b_DT=zeros(settings.n*settings.m,settings.N_mesh);
    f_inhomo=zeros(settings.n*settings.m,settings.N_mesh);
    f_DT=zeros(settings.n*settings.m,settings.N_mesh);
    f_delt=zeros(settings.n*settings.m,settings.N_mesh);
end
    
helm=helmholtz_genGPU( settings.n, settings.m);
helm.dt=settings.dt;
[settings,Mvor,Mgyro,Mlap,Rd,Rd2,Mp1,Mp3]=all_mat_gen(settings);

Mint=gpuArray(settings.Mint);
MintSq=gpuArray(settings.MintSq);

Mvor=gpuArray(Mvor);
Mgyro=gpuArray(Mgyro);
Mlap=gpuArray(Mlap);
Rd=gpuArray(Rd);
Rd2=gpuArray(Rd2);
Mp1=gpuArray(Mp1);
Mp3=gpuArray(Mp3);

f=gpuArray(f);

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
    bx_RHS,Mint,MintSq,helm,bx);
bz = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
    bz_RHS,Mint,MintSq,helm,bz);
b_DT = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
    df,Mint,MintSq,helm,b_DT);
f_inhomo = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
    inhomo_RHS,Mint,MintSq,helm,f_inhomo);
f_DT = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
    d2f,Mint,MintSq,helm,f_DT);
% f_u = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
%     U_profile(j).*dxf+W_profile.*dzf,Mint,MintSq,Mp1,Mp3,Nx_mesh,helm);
ex=gather(ex);
ez=gather(ez);
Dxx=gather(real(Mint*(Mp1*bx)*2*pi));
Dxz=gather(real(Mint*(Mp1*bz)*2*pi));
Dzx=gather(real(Mint*(Mp3*bx)*2*pi));
Dzz=gather(real(Mint*(Mp3*bz)*2*pi));
Vix=gather(real(Mint*(Mp1*f_inhomo)*2*pi));
Viz=gather(real(Mint*(Mp3*f_inhomo)*2*pi));
VDTx=gather(real(Mint*(Mp1*f_DT)*2*pi));
VDTz=gather(real(Mint*(Mp3*f_DT)*2*pi));
DDTx=gather(real(Mint*(Mp1*b_DT)*2*pi));
DDTz=gather(real(Mint*(Mp3*b_DT)*2*pi));
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

if nargin>4
    f_delt = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
        gpuArray(fdt),Mint,MintSq,helm,f_delt);
    Vdeltx = gather(real(Mint*(Mp1*f_delt)))*2*pi;
    Vdeltz = gather(real(Mint*(Mp3*f_delt)))*2*pi;
end
end


