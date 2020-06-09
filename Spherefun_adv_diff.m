clear all;

%% Setting up
dt = 0.01;                         % Time step
tfinal = 16;                        % Stopping time
nsteps = ceil(tfinal/dt);          % Number of time steps
m = 20;                            % Spatial discretization - phi
n = 20;                            % Spaptial discretization - theta
diff_const = 1;                      % Diffusion constant
beta=2.2;
S=2.5;
settings.S=S;
settings.beta=beta;
settings.n=n;
settings.m=m;
settings.omg1=1;
settings.omg2=0;
settings.omg3=0;
%% Integral weight

arr=[-n/2:n/2-1];
fac=2./(1-arr.^2);
if mod(n/2,2)
    fac(1:2:end)=0;
    fac(n/2)=0;
    fac(n/2+2)=0;
else
    fac(2:2:end)=0;
end
settings.Mint=kron(fac,[zeros(1,m/2) 1 zeros(1,m/2-1)]);
settings.MintSq=settings.Mint*settings.Mint';


%% Initial Condition
u0 = spherefun.sphharm(0,0)/sqrt(4*pi)+(spherefun.sphharm(6,0) + sqrt(14/11)*spherefun.sphharm(6,5))/sqrt(4*pi)/3;
% u0 = spherefun.sphharm(6,0) + sqrt(14/11)*spherefun.sphharm(6,5);
ucoeff=reshape(transpose(coeffs2(u0,m,n)),n*m,1);
settings.int_const=1.;
% ucoeff=zeros(n*m,1);ucoeff(m*n/2+m/2+1,1)=1/4/pi;
%% RK3 coeff and constants
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];


% Preparing Constants
K2 = (1/(dt*diff_const));         % Helmholtz frequency for BDF1


%% Initialise Recorded values
Sf=NaN(nsteps,1);
% Sf_rhs1=NaN(nsteps,1);Sf_rhs2=NaN(nsteps,1);Sf_rhs3=NaN(nsteps,1);
% Sf_lap1=NaN(nsteps,1);Sf_lap2=NaN(nsteps,1);Sf_lap3=NaN(nsteps,1);
% Sf_adv1=NaN(nsteps,1);Sf_adv2=NaN(nsteps,1);Sf_adv3=NaN(nsteps,1);
Sf(1)=int_cal(ucoeff,settings);

%% Loop!
for i = 2:nsteps
    
    k=1;
    adv_coeff=adv_cal(ucoeff,settings);
    lap_coeff=lap_cal(ucoeff,settings);
    rhs_coeff = -K2/alpha(k)*ucoeff-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*adv_coeff);
    ucoeff = helmholtz_coeff(rhs_coeff, -K2/alpha(k), n, m);
    
%     Sf_adv1(i)=int_cal(adv_coeff,settings);
%     Sf_lap1(i)=int_cal(lap_coeff,settings);
%     Sf_rhs1(i)=int_cal(rhs_coeff,settings)/(-K2/alpha(k));    
    
    k=2;
    adv_p_coeff=adv_coeff;
    adv_coeff=adv_cal(ucoeff,settings);
    lap_coeff=lap_cal(ucoeff,settings);
    rhs_coeff = -K2/alpha(k)*ucoeff-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*adv_coeff+rho(k)*adv_p_coeff);
    ucoeff = helmholtz_coeff(rhs_coeff, -K2/alpha(k), n, m);
    
%     Sf_adv2(i)=int_cal(adv_coeff,settings);
%     Sf_lap2(i)=int_cal(lap_coeff,settings);
%     Sf_rhs2(i)=int_cal(rhs_coeff,settings)/(-K2/alpha(k));    
    
    k=3;
    adv_p_coeff=adv_coeff;
    adv_coeff=adv_cal(ucoeff,settings);
    lap_coeff=lap_cal(ucoeff,settings);
    rhs_coeff = -K2/alpha(k)*ucoeff-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*adv_coeff+rho(k)*adv_p_coeff);
    rhs_coeff = rhs_coeff-settings.Mint'*(settings.Mint*rhs_coeff-settings.int_const/2/pi*(-K2/alpha(k)))/settings.MintSq;
    ucoeff = helmholtz_coeff(rhs_coeff, -K2/alpha(k), n, m);
    
%     Sf_adv3(i)=int_cal(adv_coeff,settings);
%     Sf_lap3(i)=int_cal(lap_coeff,settings);
%     Sf_rhs3(i)=int_cal(rhs_coeff,settings)/(-K2/alpha(k));
    
    
    %    Plot the solution every 25 time steps
    %     if ( mod(i, 5) == 0 )
    %         contour(u,[0:0.025:0.4]);
    %         title(sprintf('Time %2.5f',i*dt)), snapnow
    %     end
    Sf(i)=int_cal(ucoeff,settings);
    
end


%% Surface Integral Conservation check
% figure;
% plot([dt:dt:tfinal],Sf);

%% Translate back to Sphere for Post-Processing
% u=spherefun.coeffs2spherefun(transpose(reshape(ucoeff,m,n)));
% 
% n_phi=32; % Has to be even for FFT. 2^N recommended
% n_theta=101; % Had better to be 1+(multiples of 5,4,3 or 2) for Newton-Cot
% 
% dtheta=(pi/(n_theta-1));
% dphi=2*pi/(n_phi);
% 
% theta=(0:dtheta:pi)';
% phi=0:dphi:(2*pi-dphi);
% 
% 
% figure;
% contour(phi,theta,u(phi,theta));

%% Functions to generate matrices (TODO: move to function files)
function int=int_cal(ucoeff,settings)
    int=real(settings.Mint*ucoeff*2*pi);
end

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
        ,spdiags(-((-m/2:m/2-1).^2)',0, m, m));
    
    Mlap=Mtheta2+Mphi2;
end
lap=(Mlap)*ucoeff;

lap=lap-settings.Mint'*(settings.Mint*lap)/settings.MintSq;
end

function adv=adv_cal(ucoeff,settings)
persistent Madv n m
if isempty(Madv)
    n=settings.n;
    m=settings.m;
    Msinncosm = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m)); %e1
    Msinnsinm = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m)); %e2
    Mcosn = kron(spdiags(.5*ones(n,1)*[1,1], [-1 1], n, n),speye(m)); %e3
    
%     invMsinn_sinm = kron(inv(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)),spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m)); %for de1
    invMsinn_sinm = kron(inv(spdiags(ones(n,1)*[-1,1], [-1 1], n, n)),spdiags(ones(m,1)*[-1,1], [-1 1], m, m)); %for de1
%     invMsinn_cosm = kron(inv(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m)); %for de2
    invMsinn_cosm = kron(inv(spdiags(1i*ones(n,1)*[-1,1], [-1 1], n, n)),spdiags(ones(m,1)*[1,1], [-1 1], m, m)); %for de2
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
    
    adv1_coeff=settings.S/2*vor1+settings.beta*gyro1;
    adv2_coeff=settings.S/2*vor2+settings.beta*gyro2;
    adv3_coeff=settings.S/2*vor3+settings.beta*gyro3;
    %
    Madv=(Mde1*adv1_coeff+Mde2*adv2_coeff+Mde3*adv3_coeff);
    
end

adv=Madv*ucoeff;

adv=adv-settings.Mint'*(settings.Mint*adv)/settings.MintSq;

end