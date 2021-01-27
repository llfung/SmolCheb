function [settings,Mvor,Mgyro,Mlap,Rd,Rd2,Mp1,Mp3,Mp1p3,Mp3sq]=all_mat_gen(settings)

n=settings.n;
m=settings.m;
N_mesh=settings.N_mesh;

% Surface Integrals
arr=[-n/2:n/2-1];
fac=2./(1-arr.^2);
if mod(n/2,2)
    fac(1:2:end)=0;
    fac(n/2)=0;
    fac(n/2+2)=0;
else
    fac(2:2:end)=0;
end
Mint=kron(fac,[zeros(1,m/2) 1 zeros(1,m/2-1)]);
MintSq=Mint*Mint';
settings.Mint=Mint;
settings.MintSq=MintSq;

settings.Kp=settings.Kp/MintSq/settings.diff_const/settings.dt;

% Advection
Mvor=adv_vor_mat(settings)+settings.B*adv_strain_mat(settings);
Mgyro=settings.beta*adv_gyro_mat(settings);

%Laplacian
Mlap=lap_mat(settings);

Rd=spdiags(ones(N_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],N_mesh,N_mesh);
Rd=spdiags(ones(N_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-N_mesh+3:-1:-N_mesh+1 N_mesh-1:-1:N_mesh-3],Rd);
Rd=Rd/settings.d_spatial;
Rd2=spdiags(ones(N_mesh,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90],[3:-1:-3],N_mesh,N_mesh);
Rd2=spdiags(ones(N_mesh,1)*[1/90 -3/20 3/2 3/2 -3/20 1/90],[-N_mesh+3:-1:-N_mesh+1 N_mesh-1:-1:N_mesh-3],Rd2);
Rd2=Rd2/settings.d_spatial/settings.d_spatial;

%p1
Mp1 = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));
%p3
Mp3 = kron(spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n),speye(m));
%p1p3
Mp1p3 = kron(spdiags(.25i*ones(n,1)*[-1,1], [-2 2], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));
%p3p3
Mp3sq= kron(spdiags(ones(n,1)*[.25,.5,.25], [-2 0 2], n, n),speye(m));
end