%% Time marching way to invert Lp operator for transformed variables
% Note: it is found that time marching does not converge.
% Likely due to implementation of Mlap and helmholtz algorithm
% Residue modes at theta 0th, phi +-1st modes, and other phi +-1st modes.
% DO NOT USE THIS. USE DIRECT INVERSION WITH Mlap!!!!!!!
function  [ex,ez,Dxx,Dxz,Dzx,Dzz,Vix,Viz]=time_relaxed_Linv_f_zFourier(settings,S_profile,z_width,f)
%% Initialisation for faster runtime
    persistent bx bz f_inhomo 
if isempty(bx)
    bx=zeros(settings.n*settings.m,size(f,2));
    bz=zeros(settings.n*settings.m,size(f,2));
    f_inhomo=zeros(settings.n*settings.m,size(f,2)); 
end
%% Initialisation    
helm=helmholtz_genGPU( settings.n, settings.m);
helm.dt=settings.dt;
[settings,Mvor,Mgyro,Mlap,Rd,~,Mp1,Mp3]=all_mat_gen(settings);

Mint=gpuArray(settings.Mint);
MintSq=gpuArray(settings.MintSq);

Mvor=gpuArray(Mvor);
Mgyro=gpuArray(Mgyro);
Mlap=gpuArray(Mlap);

Nz_mesh=size(f,2)/settings.N_mesh;
Rdx=gpuArray(kron(speye(Nz_mesh),Rd));
Rd=spdiags(ones(Nz_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],Nz_mesh,Nz_mesh);
Rd=spdiags(ones(Nz_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-Nz_mesh+3:-1:-Nz_mesh+1 Nz_mesh-1:-1:Nz_mesh-3],Rd);
Rd=Rd/(z_width/Nz_mesh);
Rdz=gpuArray(kron(Rd,speye(settings.N_mesh)));
%Rd2=gpuArray(Rd2);
Mp1=gpuArray(Mp1);
Mp3=gpuArray(Mp3);

f=gpuArray(f);
%% Computation
dfz=f*Rdz;
dfx=f*Rdx;

ex=real(Mint*Mp1*f*(2*pi));
ez=real(Mint*Mp3*f*(2*pi));

bx_RHS=Mp1*f-ex.*f;
bz_RHS=Mp3*f-ez.*f;


inhomo_RHS=Mp1*(dfx)+Mp3*(dfz)-(ex*Rdx+ez*Rdz).*f;

bx = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
    bx_RHS,Mint,MintSq,helm,bx);
bz = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
    bz_RHS,Mint,MintSq,helm,bz);
f_inhomo = time_relaxed_Linv(Mvor,Mgyro,Mlap,S_profile,...
    inhomo_RHS,Mint,MintSq,helm,f_inhomo);

ex=gather(ex);
ez=gather(ez);
Dxx=gather(real(Mint*(Mp1*bx)*2*pi));
Dxz=gather(real(Mint*(Mp1*bz)*2*pi));
Dzx=gather(real(Mint*(Mp3*bx)*2*pi));
Dzz=gather(real(Mint*(Mp3*bz)*2*pi));
Vix=gather(real(Mint*(Mp1*f_inhomo)*2*pi));
Viz=gather(real(Mint*(Mp3*f_inhomo)*2*pi));

end


