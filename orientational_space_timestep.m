% Timestepping relaxation to solve for f and b in spherical coordinate
clear all
% c = parcluster;
% c.NumWorkers = 32;
% saveProfile(c);
% parpool(32);

zi=sqrt(-1);

%% Define discretization parameters
settings.n_phi=48;      % Has to be even for FFT. 2^N recommended
settings.n_theta=101;   % Had better to be 1+(multiples of 5,4,3 or 2) for Newton-Cote
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
% G= (\nabla V)^T, V=horizontal vector
% Gij=\partial_1 V_j
settings.G11=0;
settings.G12=1;  %dU/dr
settings.G13=0;
settings.G21=0;
settings.G22=0;
settings.G23=0;
settings.G31=0;
settings.G32=0;
settings.G33=0;
settings.omega_e1=0;
settings.omega_e2=0;
settings.omega_e3=1;

%% Initial Condition
fi=ones(settings.n_theta-2,settings.n_phi)/4/pi;
fi_col=reshape(fi,(settings.n_theta-2)*settings.n_phi,1);

bi=zeros(2*settings.n_theta-4,settings.n_phi);
bi_col=reshape(bi,2*(settings.n_theta-2)*settings.n_phi,1);

%% Jacoabian for run accelerations
J = kron( spdiags(ones(settings.n_theta-2,3),-1:1,...
    settings.n_theta-2,settings.n_theta-2),...
    ones(settings.n_phi,settings.n_phi));

% full Jb
% Jb= kron( spdiags(ones(3*(settings.n_theta-2),3),-1:1,...
%     3*(settings.n_theta-2),3*(settings.n_theta-2)),...
%     ones(settings.n_phi,settings.n_phi))+...
%     kron( ones(3,3),...
%     eye((settings.n_theta-2)*settings.n_phi));


%% Initialisation
settings.K_p=10;
% settings.bK_p=1;
S_loop=[0:10];
settings.S=2.5;

%% Loop for different S
N_loop=length(S_loop);
for i=1:N_loop
    %% Set up parameters in loop
    settings_loc=settings;
    settings_loc.S=S_loop(i);
    
    %% Solving for f(e)
    tic
    %opt=odeset('RelTol',1e-4,'AbsTol',1e-5,'JPattern',S,'NormControl','off','Events',@fkode_f_event);
    opt=odeset('RelTol',1e-4/3,'AbsTol',1e-13,'JPattern',J,'NormControl','off','MaxStep',0.1);
    [t_sol,f_sol]=ode23t(@(t,y) fkode_cyl(t,y,settings_loc),t_default,fi_col,opt);

        f=transpose(reshape(f_sol(end,:),settings_loc.n_phi,settings_loc.n_theta-2));
        f_theta_0=ones(1,settings_loc.n_phi)*(mean(4*f(2,:))-mean(f(3,:)))/3;
        f_theta_n=ones(1,settings_loc.n_phi)*mean(4*f(end-1,:)-f(end-2,:))/3;
        f0=[f_theta_0;f;f_theta_n];settings_loc.f0=f0;
    toc
    

    disp([num2str(i) '/' num2str(N_loop)]);
end

%% Saving
% name=['beta_' num2str(settings.beta*10) 'G12'];
% save([name '.mat']);
% exit
