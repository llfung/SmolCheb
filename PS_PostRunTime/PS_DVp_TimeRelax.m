addpath(genpath('core'))
addpath(genpath('core/x_FD'))
addpath(genpath('core/p_DFS'))
addpath(genpath('core/PS_RT'))
dir='./';

load([dir 'Summary.mat']);

[settings,Mvor,Mgyro,Mlap,Rdx,Rd2x,Mp1,Mp3,Mp1p3,~]=all_mat_gen(settings);
helm=helmholtz_genGPU( settings.n, settings.m);
helm.dt=settings.dt*10;

g_trelax=time_relaxed_Linv(gpuArray(Mvor),gpuArray(Mgyro),gpuArray(Mlap),gpuArray(S_profile),...
zeros(n*m,N_mesh,'gpuArray'),gpuArray(settings.Mint),gpuArray(settings.MintSq),helm);

[Trans_trelax.ex_g,Trans_trelax.ez_g,Trans_trelax.Dxx_g,Trans_trelax.Dxz_g,Trans_trelax.Dzx_g,Trans_trelax.Dzz_g,Trans_trelax.Vix_g,Trans_trelax.Viz_g,...
        Trans_trelax.VDTx_g,Trans_trelax.VDTz_g,Trans_trelax.DDTx_g,Trans_trelax.DDTz_g]...
        =time_relaxed_Linv_f('x',settings,S_profile,g_trelax);
%%
for i=1:length(t2)
    load([dir 't' num2str(t2(i)) '.mat']);
    f=ufull_save./real(settings.Mint*ufull_save*2*pi);
    
    [ex,ez,Dxx,Dxz,Dzx,Dzz,Vix,Viz,...
        VDTx,VDTz,DDTx,DDTz,Vdeltx,Vdeltz]...
        =time_relaxed_Linv_f('x',settings,S_profile,f,fdt_full_save);
    
    Trans_trelax.ex(i,:)=ex;
    Trans_trelax.ez(i,:)=ez;
    Trans_trelax.Dxx(i,:)=Dxx;
    Trans_trelax.Dxz(i,:)=Dxz;
    Trans_trelax.Dzx(i,:)=Dzx;
    Trans_trelax.Dzz(i,:)=Dzz;
    Trans_trelax.Vix(i,:)=Vix;
    Trans_trelax.Viz(i,:)=Viz;
    Trans_trelax.VDTx(i,:)=VDTx;
    Trans_trelax.VDTz(i,:)=VDTz;
    Trans_trelax.DDTxx(i,:)=DDTx;
    Trans_trelax.DDTzx(i,:)=DDTz;
%     Trans_trelax.DDTxz(i,:)=DDTx;
%     Trans_trelax.DDTzz(i,:)=DDTz;
    Trans_trelax.Vux(i,:)=Vdeltx;
    Trans_trelax.Vuz(i,:)=Vdeltz;
    disp([num2str(i) '/' num2str(length(t2))]);
end
%%
save([dir 'SummaryPS.mat']);
