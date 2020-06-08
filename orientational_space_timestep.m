% Timestepping relaxation to solve for f and b in spherical coordinate
clear all
% c = parcluster;
% c.NumWorkers = 32;
% saveProfile(c);
% parpool(32);

zi=sqrt(-1);

%% Define discretization parameters
settings.n_phi=16;      % Has to be even for FFT. 2^N recommended
settings.n_theta=21;   % Had better to be 1+(multiples of 5,4,3 or 2) for Newton-Cote
settings.tn_phi=settings.n_phi/2+1;
settings.tn_theta=(floor(settings.n_theta/2)+1);

settings.dtheta=(pi/(settings.n_theta-1));
settings.dphi=2*pi/(settings.n_phi);

settings.theta=[0:settings.dtheta:pi]';
settings.phi=[0:settings.dphi:(2*pi-settings.dphi)];
settings.kphi=[0:(settings.tn_phi-1)];
settings.e1=sin(settings.theta)*cos(settings.phi);
settings.e2=cos(settings.theta)*ones(size(settings.phi));
settings.e3=sin(settings.theta)*sin(settings.phi);

t_default=[0 16];

endings=[1 settings.n_theta-2 settings.n_theta-1 2*settings.n_theta-4 2*settings.n_theta-3 3*settings.n_theta-6];
%% Define input parameters
settings.S=2.5;
settings.beta=2.2;

settings.omega_e1=0;
settings.omega_e2=0;
settings.omega_e3=1;

%% Initial Condition
fi=ones(settings.n_theta-2,settings.n_phi)/4/pi;
fi_col=reshape(fi,(settings.n_theta-2)*settings.n_phi,1);

% fi_col=[fi_col;0];
% fpi_col=orient_op(0,fi_col,settings);
%% Jacoabian for run accelerations
J = kron( spdiags(ones(settings.n_theta-2,3),-1:1,...
    settings.n_theta-2,settings.n_theta-2),...
    ones(settings.n_phi,settings.n_phi));

% Jf=[J zeros(settings.n_phi*(settings.n_theta-2),1);ones(1,settings.n_phi*(settings.n_theta-2)) 0];
% 
% M=speye(settings.n_phi*(settings.n_theta-2)+1);
% M(end,end)=0;
%% Initialisation
settings.K_p=10;
    
%% Solving for f(e)
tic
% opt=odeset('RelTol',1e-4,'AbsTol',1e-5,'JPattern',S,'NormControl','off','Events',@fkode_f_event);
 opt=odeset('RelTol',1e-4/3,'AbsTol',1e-13,'JPattern',J,'NormControl','off','MaxStep',0.1);
[t_sol,f_sol]=ode23t(@(t,y) orient_op(t,y,settings),t_default,fi_col,opt);

    f=transpose(reshape(f_sol(end,:),settings.n_phi,settings.n_theta-2));
    f_theta_0=ones(1,settings.n_phi)*(mean(4*f(1,:))-mean(f(2,:)))/3;
    f_theta_n=ones(1,settings.n_phi)*mean(4*f(end,:)-f(end-1,:))/3;
    f0=[f_theta_0;f;f_theta_n];
toc

    Sf=NaN(size(t_sol));
for j=1:length(t_sol)
    f=transpose(reshape(f_sol(j,:),settings.n_phi,settings.n_theta-2));
    f_theta_0=ones(1,settings.n_phi)*(mean(4*f(1,:))-mean(f(2,:)))/3;
    f_theta_n=ones(1,settings.n_phi)*mean(4*f(end,:)-f(end-1,:))/3;
    f0=[f_theta_0;f;f_theta_n];

    cf= forward_fft(f0);
    Sf(j)=area(cf,settings.theta);
end


%% Saving
% name=['beta_' num2str(settings.beta*10) 'G12'];
% save([name '.mat']);
% exit
