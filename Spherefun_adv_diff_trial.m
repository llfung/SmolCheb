%% Initial Condition
u0 = spherefun.sphharm(0,0)/sqrt(4*pi)+(spherefun.sphharm(6,0) + sqrt(14/11)*spherefun.sphharm(6,5))/sqrt(4*pi)/3;
% u0 = spherefun.sphharm(6,0) + sqrt(14/11)*spherefun.sphharm(6,5);
%  u0 = spherefun(@(x,y,z)1/4/pi,[m m]);

omg=spherefunv(0,1,0);
kvec=spherefunv(0,0,1);
e=spherefunv(@(x,y,z)x,@(x,y,z)y,@(x,y,z)z);
% plot(u0), colormap(flipud(hot)), caxis([-1 1.5]), colorbar, axis('off')
%% Setting up
dt = 0.01;                         % Time step
tfinal = 3;                        % Stopping time
nsteps = ceil(tfinal/dt);          % Number of time steps
m = 20;                            % Spatial discretization
diff_const = 1;                      % Diffusion constant
beta=2.2;
S=2.5;

u=u0;
zero_spherefun=spherefun(@(x,y,z)0);
z_spherefun=spherefun(@(x,y,z)z);

tic
gyro_term_precompute=kvec-times(e,z_spherefun);
vorticity_term_precompute=cross(omg,e);
adv_pre=beta*gyro_term_precompute+S/2*vorticity_term_precompute;
adv_cells=adv_pre.components;
adv1_coeff=coeffs2(adv_cells{1},m,m);
adv2_coeff=coeffs2(adv_cells{2},m,m);
adv3_coeff=coeffs2(adv_cells{3},m,m);
whole_term_pre1=spherefun.coeffs2spherefun(adv1_coeff);
whole_term_pre2=spherefun.coeffs2spherefun(adv2_coeff);
whole_term_pre3=spherefun.coeffs2spherefun(adv3_coeff);
whole_term_pre=vertcat(whole_term_pre1, whole_term_pre2, whole_term_pre3);
toc
%% RK3 coeff
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];

Sf=NaN(nsteps,1);
% Preparing Constants
K2 = (1/(dt*diff_const));         % Helmholtz frequency for BDF1
Ki = sqrt(1/(dt*diff_const))*1i;
Sf(1)=mean2(u);
tic
for n = 2:nsteps

        k=1;
        adv=div(times(whole_term_pre,u));
        rhs = -K2/alpha(k)*u-laplacian(u)+1/diff_const/alpha(k)*(gamma(k)*adv);
        u = spherefun.helmholtz(rhs, Ki/sqrt(alpha(k)), m, m);
        
        k=2;
        adv_p=adv;
        adv=div(times(whole_term_pre,u));
        rhs = -K2/alpha(k)*u-laplacian(u)+1/diff_const/alpha(k)*(gamma(k)*adv+rho(k)*adv_p);
        u = spherefun.helmholtz(rhs, Ki/sqrt(alpha(k)), m, m);
        
        k=3;
        adv_p=adv;
        adv=div(times(whole_term_pre,u));
        rhs = -K2/alpha(k)*u-laplacian(u)+1/diff_const/alpha(k)*(gamma(k)*adv+rho(k)*adv_p);
        u = spherefun.helmholtz(rhs, Ki/sqrt(alpha(k)), m, m);

%    Plot the solution every 25 time steps
%     if ( mod(n, 5) == 0 )
%         contour(u,[0:0.025:0.4]); 
%         title(sprintf('Time %2.5f',n*dt)), snapnow
%     end
    Sf(n)=mean2(u);
end
toc

%% Surface Integral Conservation check
% figure;
% plot([dt:dt:tfinal],Sf);