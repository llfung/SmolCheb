%% Time marching way to invert Lp operator for Transformed variables
% Note: it is found that time marching does not converge.
% Likely due to implementation of Mlap and helmholtz algorithm
% Residue modes at theta 0th, phi +-1st modes, and other phi +-1st modes.


% load('/mnt/d/Sedimentation/20220913_spheroidB0.31_PesLarge/Pes1_Ri45_DT0.01/smolbuoy_pBC_0beta_0DT_0Vc_10AR_0.31B_0M_1Vs_1Re_0.01Pef_45Ri_cospi_128cd_8m_12n_0.0005dt_0ti_100.001tf.mat')
load('smolbuoy_pBC_0beta_0DT_0Vc_10AR_0.31B_0M_1Vs_1Re_0.01Pef_45Ri_cospi_128cd_8m_12n_0.0005dt_0ti_100.001tf.mat')
job_id=str2num(getenv('SLURM_ARRAY_TASK_ID'));
% job_id=4;
addpath(genpath('core'))
addpath(genpath('core/x_FD'))
addpath(genpath('core/p_DFS'))
addpath(genpath('core/PS_RT'))
% dir='/mnt/d/Sedimentation/20220913_spheroidB0.31_PesLarge/Pes1_Ri45_DT0.01/';
dir='./';

% load([dir 'Summary.mat']);
[settings,Mvor,Mgyro,Minert,Mlap,Rdx,Rd2x,Mp1,Mp3,Mp1p3,~]=all_mat_gen(settings);

helm=helmholtz_genGPU( settings.n, settings.m);
helm.dt=settings.dt*1;

% Msin=kron(spdiags(0.5i*ones(settings.n,1)*[-1,1], [-1 1], settings.n, settings.n),speye(settings.m));
% [g,Linv]=Linv_g(S_profile,Mvor,Mgyro,Mlap,settings.Mint,Msin,n,m);
% [Transformed.ex_g,Transformed.ez_g,Transformed.Dxx_g,Transformed.Dxz_g,Transformed.Dzx_g,Transformed.Dzz_g,...
% Transformed.Vix_g,Transformed.Viz_g,Transformed.VDTx_g,Transformed.VDTz_g,Transformed.DDTxx_g,Transformed.DDTzx_g]=...
% Linv_f('x',g,Linv,Msin,Rdx,Rd2x,Mp1,Mp3,settings,zeros(1,N_mesh),n*m/2+m/2+1);

Trans_trelax.ex_g=NaN(length(t2),N_mesh);
Trans_trelax.ez_g=NaN(length(t2),N_mesh);
Trans_trelax.Dxx_g=NaN(length(t2),N_mesh);
Trans_trelax.Vix_g=NaN(length(t2),N_mesh);
Trans_trelax.Viz_g=NaN(length(t2),N_mesh);
Trans_trelax.VDTx_g=NaN(length(t2),N_mesh);
Trans_trelax.VDTz_g=NaN(length(t2),N_mesh);
Trans_trelax.DDTxx_g=NaN(length(t2),N_mesh);
Trans_trelax.DDTzx_g=NaN(length(t2),N_mesh);

Trans_trelax.ex=NaN(length(t2),N_mesh);
Trans_trelax.ez=NaN(length(t2),N_mesh);
Trans_trelax.Dxx=NaN(length(t2),N_mesh);
Trans_trelax.Vix=NaN(length(t2),N_mesh);
Trans_trelax.Viz=NaN(length(t2),N_mesh);
Trans_trelax.VDTx=NaN(length(t2),N_mesh);
Trans_trelax.VDTz=NaN(length(t2),N_mesh);
Trans_trelax.DDTxx=NaN(length(t2),N_mesh);
Trans_trelax.DDTzx=NaN(length(t2),N_mesh);

%%
g_trelax=zeros(settings.n*settings.m,N_mesh);
g_trelax(helm.n*helm.m/2+helm.n/2+1,:)=1/4/pi;
for i=(job_id-1)*1000+1:job_id*1000
    load([dir 't' num2str(t2(i)) '.mat']);
    f=ufull_save./real(settings.Mint*ufull_save*2*pi);
    S_profile=Wfull_save*Rdx;
    
    g_trelax=time_relaxed_Linv(gpuArray(Mvor),gpuArray(Mgyro),gpuArray(Mlap),gpuArray(S_profile),...
zeros(n*m,N_mesh,'gpuArray'),gpuArray(settings.Mint),gpuArray(settings.MintSq),helm,g_trelax);

    [ex_g,ez_g,Dxx_g,Vix_g,Viz_g,VDTx_g,VDTz_g,DDTx_g,DDTz_g]...
        =time_relaxed_Linv_f_sedi('x',settings,S_profile,g_trelax);

    Trans_trelax.ex_g(i,:)=ex_g;
    Trans_trelax.ez_g(i,:)=ez_g;
    Trans_trelax.Dxx_g(i,:)=Dxx_g;
%     Trans_trelax.Dxz_g(i,:)=Dxz_g;
%     Trans_trelax.Dzx_g(i,:)=Dzx_g;
%     Trans_trelax.Dzz_g(i,:)=Dzz_g;
    Trans_trelax.Vix_g(i,:)=Vix_g;
    Trans_trelax.Viz_g(i,:)=Viz_g;
    Trans_trelax.VDTx_g(i,:)=VDTx_g;
    Trans_trelax.VDTz_g(i,:)=VDTz_g;
    Trans_trelax.DDTxx_g(i,:)=DDTx_g;
    Trans_trelax.DDTzx_g(i,:)=DDTz_g;

    [ex,ez,Dxx,Vix,Viz,VDTx,VDTz,DDTx,DDTz]...
        =time_relaxed_Linv_f_sedi('x',settings,S_profile,f);
    
    Trans_trelax.ex(i,:)=ex;
    Trans_trelax.ez(i,:)=ez;
    Trans_trelax.Dxx(i,:)=Dxx;
%     Trans_trelax.Dxz(i,:)=Dxz;
%     Trans_trelax.Dzx(i,:)=Dzx;
%     Trans_trelax.Dzz(i,:)=Dzz;
    Trans_trelax.Vix(i,:)=Vix;
    Trans_trelax.Viz(i,:)=Viz;
    Trans_trelax.VDTx(i,:)=VDTx;
    Trans_trelax.VDTz(i,:)=VDTz;
    Trans_trelax.DDTxx(i,:)=DDTx;
    Trans_trelax.DDTzx(i,:)=DDTz;
%     Trans_trelax.DDTxz(i,:)=DDTx;
%     Trans_trelax.DDTzz(i,:)=DDTz;
%     Trans_trelax.Vux(i,:)=Vdeltx;%     Trans_trelax.Vuz(i,:)=Vdeltz;
    disp([num2str(i) '/' num2str(length(t2))]);
end
%%
% clearvars Linv;
save([dir 'SummaryPS_' num2str(job_id) '.mat']);
