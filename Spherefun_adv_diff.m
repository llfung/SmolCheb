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
tfinal = 2;                        % Stopping time
nsteps = ceil(tfinal/dt);          % Number of time steps
m = 20;                            % Spatial discretization
diff_const = 1;                      % Diffusion constant
beta=2.2;
S=2.5;
settings.S=S;
settings.beta=beta;
settings.n=m;
settings.m=m;n=m;
settings.omg1=0;
settings.omg2=1;
settings.omg3=0;

u=u0;

%% RK3 coeff
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];

Sf=NaN(nsteps,1);
Sf_adv=NaN(nsteps,1);
% Preparing Constants
K2 = (1/(dt*diff_const));         % Helmholtz frequency for BDF1
Ki = sqrt(1/(dt*diff_const))*1i;
Sf(1)=mean2(u);
% Sf_adv(1)=0;
ucoeff=reshape(transpose(coeffs2(u,m,n)),n*m,1);
tic
for i = 2:nsteps

        k=1;
        adv=adv_cal(ucoeff,settings);
        rhs_coeff = -K2/alpha(k)*ucoeff-lap_cal(ucoeff,settings)+1/diff_const/alpha(k)*(gamma(k)*adv);
        ucoeff = helmholtz_coeff(rhs_coeff, Ki/sqrt(alpha(k)), m, m);

        
        k=2;
        adv_p=adv;
        adv=adv_cal(ucoeff,settings);
        rhs_coeff = -K2/alpha(k)*ucoeff-lap_cal(ucoeff,settings)+1/diff_const/alpha(k)*(gamma(k)*adv+rho(k)*adv_p);
        ucoeff = helmholtz_coeff(rhs_coeff, Ki/sqrt(alpha(k)), m, m);

        
        k=3;
        adv_p=adv;
        adv=adv_cal(ucoeff,settings);
        rhs_coeff = -K2/alpha(k)*ucoeff-lap_cal(ucoeff,settings)+1/diff_const/alpha(k)*(gamma(k)*adv+rho(k)*adv_p);
        ucoeff = helmholtz_coeff(rhs_coeff, Ki/sqrt(alpha(k)), m, m);


%    Plot the solution every 25 time steps
%     if ( mod(i, 5) == 0 )
%         contour(u,[0:0.025:0.4]); 
%         title(sprintf('Time %2.5f',i*dt)), snapnow
%     end
    Sf(i)=mean2(spherefun.coeffs2spherefun(transpose(reshape(ucoeff,m,n))));
%     Sf_adv(i)=mean2(adv);
%     Sf(i)=mean2(u);

end
toc

%% Surface Integral Conservation check
% figure;
% plot([dt:dt:tfinal],Sf);
% 
function lap=lap_cal(ucoeff,settings)
persistent Mlap n m
if isempty(Mlap)
n=settings.n;
m=settings.m;
Mtheta2 = kron((spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n))\...
    (spdiags((-n/2:n/2-1)'*1i,0, n, n)*...
    spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)*...
    spdiags((-n/2:n/2-1)'*1i,0, n, n)),speye(m));

Mphi2 = kron((spdiags(.25*ones(n,1)*[-1,2,-1], [-2:2:2], n, n)\speye(n))...
    ,spdiags(-((-n/2:n/2-1).^2)',0, n, n));

Mlap=Mtheta2+Mphi2;
end
lap=(Mlap)*ucoeff;
end

function adv=adv_cal(ucoeff,settings)
persistent Madv n m
if isempty(Madv)
n=settings.n;
m=settings.m;
Msinncosm = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m)); %e1
Msinnsinm = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m)); %e2
Mcosn = kron(spdiags(.5*ones(n,1)*[1,1], [-1 1], n, n),speye(m)); %e3

invMsinn_sinm = kron(inv(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)),spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m)); %for de1
invMsinn_cosm = kron(inv(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m)); %for de2
Mde1= -invMsinn_sinm*kron(speye(n),spdiags((-m/2:m/2-1)'*1i,0, m, m))...
    + kron(spdiags(.5*ones(n,1)*[1,1], [-1 1], n, n)*spdiags((-n/2:n/2-1)'*1i,0, n, n),...
    spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m)); % de1
Mde2= invMsinn_cosm*kron(speye(n),spdiags((-m/2:m/2-1)'*1i,0, m, m))...
    + kron(spdiags(.5*ones(n,1)*[1,1], [-1 1], n, n)*spdiags((-n/2:n/2-1)'*1i,0, n, n),...
    spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m)); % de2
Mde3= -kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)*spdiags((-n/2:n/2-1)'*1i,0, n, n),speye(m)); % de3



vor1=(settings.omg2*Mcosn    -settings.omg3*Msinnsinm);
vor2=(settings.omg3*Msinncosm-settings.omg1*Mcosn    );
vor3=(settings.omg1*Msinnsinm-settings.omg2*Msinncosm);

gyro1=-Mcosn*Msinncosm;
gyro2=-Mcosn*Msinnsinm;
gyro3=(speye(n*m)-Mcosn*Mcosn);

% adv1_coeff=transpose(reshape(settings.S/2*vor1+settings.beta*gyro1,m,n));
% adv2_coeff=transpose(reshape(settings.S/2*vor2+settings.beta*gyro2,m,n));
% adv3_coeff=transpose(reshape(settings.S/2*vor3+settings.beta*gyro3,m,n));
% 
% whole_term1=spherefun.coeffs2spherefun(adv1_coeff);
% whole_term2=spherefun.coeffs2spherefun(adv2_coeff);
% whole_term3=spherefun.coeffs2spherefun(adv3_coeff);
% whole_term=vertcat(whole_term1, whole_term2, whole_term3);
% 
% adv=div(whole_term);


adv1_coeff=settings.S/2*vor1+settings.beta*gyro1;
adv2_coeff=settings.S/2*vor2+settings.beta*gyro2;
adv3_coeff=settings.S/2*vor3+settings.beta*gyro3;
% 
Madv=(Mde1*adv1_coeff+Mde2*adv2_coeff+Mde3*adv3_coeff);

end

% ucoeff=reshape(transpose(coeffs2(u,m,n)),n*m,1);

% 
% whole_term1=spherefun.coeffs2spherefun(transpose(reshape(adv1_coeff*ucoeff,m,n)));
% whole_term2=spherefun.coeffs2spherefun(transpose(reshape(adv2_coeff*ucoeff,m,n)));
% whole_term3=spherefun.coeffs2spherefun(transpose(reshape(adv3_coeff*ucoeff,m,n)));
% whole_term=vertcat(whole_term1, whole_term2, whole_term3);
% 
% adv2=div(whole_term);

adv=Madv*ucoeff;

% div_coeff=transpose(reshape(...
%     Madv*ucoeff,...
%     m,n));
% adv=spherefun.coeffs2spherefun(div_coeff);
end