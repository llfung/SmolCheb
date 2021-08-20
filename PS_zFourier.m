addpath(genpath('core'))
addpath(genpath('core/x_FD'))
addpath(genpath('core/p_DFS'))
addpath(genpath('core/PS_RT'))


load('smol_pBC_2D_0.008epsInit_0beta_0B_0Vsm_0Vsv_0.0025Vc_0DT_1Pef_homo_dx_256dz_512_x2_z4_m12_n16_dt0.001_tf6.002.mat')

ucoeff_phy=ifft(reshape(ucoeff,n*m,Nx_mesh,Nz_mesh),[],3);
dz=z_width/(Nz_mesh);
f=ucoeff_phy./pagemtimes(settings.Mint,ucoeff_phy);
[ex,ez,Dxx,Dxz,Dzx,Dzz,Vix,Viz]=time_relaxed_Linv_f_zFourier(settings,S_profile,z_width,reshape(f,n*m,[]));

save('smol_pBC_2D_0.008epsInit_0beta_0B_0Vsm_0Vsv_0.0025Vc_0DT_1Pef_homo_dx_256dz_512_x2_z4_m12_n16_dt0.001_tf6.002.mat','-v7.3');
