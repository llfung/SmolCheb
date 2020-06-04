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

settings.N_mesh=5;

t_default=[0 16];

endings=[1 settings.n_theta-2 settings.n_theta-1 2*settings.n_theta-4 2*settings.n_theta-3 3*settings.n_theta-6];
%% Define input parameters
settings.S=2.5;
settings.beta=2.2;

settings.omega_e1=0;
settings.omega_e2=0;
settings.omega_e3=1;

%% Initial Condition
fi=ones(settings.n_phi,settings.n_theta-2,settings.N_mesh)/8/pi;
fi_col=reshape(fi,settings.n_phi*(settings.n_theta-2)*settings.N_mesh,1);

%% Jacoabian for run accelerations
% J = kron( sparse(ones(settings.N_mesh,settings.N_mesh)),...
%     kron( spdiags(ones(settings.n_theta-2,3),-1:1,...
%     settings.n_theta-2,settings.n_theta-2),...
%     ones(settings.n_phi,settings.n_phi))...
%     );
J = kron( spdiags(ones(settings.N_mesh,7),-3:3,settings.N_mesh,settings.N_mesh),...
    kron( spdiags(ones(settings.n_theta-2,5),-2:2,...
    settings.n_theta-2,settings.n_theta-2),...
    ones(settings.n_phi,settings.n_phi))...
    );
%% Initialisation
settings.K_p=100;

settings.S=2.5;

%% Set up parameters in loop
settings_loc=settings;

    
%% Solving for f(e)
tic
%opt=odeset('RelTol',1e-4,'AbsTol',1e-5,'JPattern',S,'NormControl','off','Events',@fkode_f_event);
opt=odeset('RelTol',1e-4/3,'AbsTol',1e-13,'JPattern',J,'NormControl','off','MaxStep',0.1);
[t_sol,f_sol]=ode23t(@(t,y) full_phy_rep_y_op(t,y,settings_loc),t_default,fi_col,opt);

    f_wobc=transpose(reshape(f_sol(end,:),settings_loc.n_phi,settings.N_mesh*(settings_loc.n_theta-2)));
    n_theta_wobc=(settings.n_theta-2);
    %% 2nd order BC implementation
    f=NaN(settings.n_theta,settings.n_phi,settings.N_mesh);
    for i=1:settings.N_mesh
        f_theta_0=ones(1,settings.n_phi)*mean(18*f_wobc((i-1)*n_theta_wobc+1,:)-9*f_wobc((i-1)*n_theta_wobc+2,:)+2*f_wobc((i-1)*n_theta_wobc+3,:))/11;
        f_theta_n=ones(1,settings.n_phi)*mean(18*f_wobc(    i*n_theta_wobc-1,:)-9*f_wobc(    i*n_theta_wobc-2,:)+2*f_wobc(    i*n_theta_wobc-3,:))/11;
        f(:,:,i)=[f_theta_0;f_wobc((i-1)*n_theta_wobc+1:i*n_theta_wobc,:);f_theta_n];
    end
    
toc
    Sf=NaN(size(t_sol));
    for j=1:length(t_sol)
        f_wobc=transpose(reshape(f_sol(j,:),settings_loc.n_phi,settings.N_mesh*(settings_loc.n_theta-2)));
        f_theta_0=ones(1,settings.n_phi)*mean(18*f_wobc((i-1)*n_theta_wobc+1,:)-9*f_wobc((i-1)*n_theta_wobc+2,:)+2*f_wobc((i-1)*n_theta_wobc+3,:))/11;
        f_theta_n=ones(1,settings.n_phi)*mean(18*f_wobc(    i*n_theta_wobc-1,:)-9*f_wobc(    i*n_theta_wobc-2,:)+2*f_wobc(    i*n_theta_wobc-3,:))/11;
        f_t=[f_theta_0;f_wobc((i-1)*n_theta_wobc+1:i*n_theta_wobc,:);f_theta_n];
        
        cf= forward_fft(f_t);
        Sf(j)=area(cf,settings.theta);
    end

%% Saving
% name=['beta_' num2str(settings.beta*10) 'G12'];
% save([name '.mat']);
% exit
