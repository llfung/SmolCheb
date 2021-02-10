%% Time Stepping solver for the Full Smoluchowski Equation (RK3-CN2)
% Loosely based on Spherefun, this solve uses Double Fourier Sphere (DFS)
% method to tranform the orientational space and time-marched in the DFS 
% space. Time stepping is semi-implicit, with Advection terms in RK3 and 
% Laplacian term in CN2. Implicit matrix inversion is done using the
% Spherefun Helmholtz Solver (only slightly modified to remove
% chebfun-based activiities). 

%% Setting up
%Saving to settings struct
settings.beta=beta;
settings.B=B;
settings.Vc=Vc;
settings.n=n;
settings.m=m;
settings.diff_const=diff_const;
settings.dt=dt;
settings.d_spatial=dx;
settings.N_mesh=N_mesh;
settings.Kp=Kp;
settings.nsteps=nsteps;

settings.omg1=G(2,3)-G(3,2);
settings.omg2=G(3,1)-G(1,3);
settings.omg3=G(1,2)-G(2,1);
settings.e11=G(1,1);
settings.e12=G(1,2)+G(2,1);
settings.e13=G(3,1)+G(1,3);
settings.e22=G(2,2);
settings.e23=G(2,3)+G(3,2);
settings.e33=G(3,3);
settings.int_const=int_const;

% RK3 coeff and constants
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];

% Preparing Constants
K2 = (1/(dt*diff_const));         % Helmholtz frequency for BDF1

%% Initialising Matrices
[settings,Mvor,Mgyro,Mlap,Mlap_inv,Rdx,Rd2x,Mp1,Mp3,Mp1p3,~]=all_mat_gen(settings);

Mint=settings.Mint;
MintSq=settings.MintSq;

Kp=settings.Kp;

mats=struct('Mint',Mint,'S_profile',S_profile,'Mvor',Mvor,'Mgyro',Mgyro,'Mlap',Mlap_inv,...
    'Mp1',Mp1,'Mp3',Mp3,'Rdx',Rdx,'Rd2x',Rd2x);

helm=helmholtz_gen( n, m);

%Swimming and sedimentation   
MSwim=Vc*Mp1-Vsvar*Mp1p3;

%% Initialise Recorded values
cell_den=NaN(floor(nsteps/saving_rate3),N_mesh);

PS=PS_RunTime('x','inv',mats,settings,saving_rate1,saving_rate2);

%% Time-Stepping (RK3-CN2)
ucoeff=ucoeff0;
adv_p_coeff     =zeros(n*m,N_mesh);
adv_comb_coeff  =zeros(n*m,N_mesh);
ucoeff_previous2=NaN(n*m,N_mesh,3);

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dx;
    
for i = 1:nsteps
    %% RK step 1
    k=1;
    % Par-For Version
    dxu_coeff=ucoeff*Rdx;
    dx2u_coeff=ucoeff*Rd2x;
    parfor j=1:N_mesh
        adv_coeff=S_profile(j)*(Mvor*ucoeff(:,j))+Mgyro*ucoeff(:,j);
        adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
        
        lap_coeff=Mlap*ucoeff(:,j);
        lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
        
        swim_coeff=MSwim*dxu_coeff(:,j);
        
        DT_coeff=DT*dx2u_coeff(:,j);
        
        adv_p_coeff(:,j)=adv_coeff+swim_coeff-DT_coeff;
        
        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_p_coeff(:,j)))...
            -Kp/alpha(k)*(int_const-Nint_loc)*Mint'.*ucoeff(:,j);
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
    end
    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dx;
    
    %% RK step 2
    k=2;
    dxu_coeff=ucoeff*Rdx;
    dx2u_coeff=ucoeff*Rd2x;
    parfor j=1:N_mesh
%         adv_p_coeff=adv_coeff+swim_coeff;
        adv_coeff=S_profile(j)*(Mvor*ucoeff(:,j))+Mgyro*ucoeff(:,j);
        adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
        
        lap_coeff=Mlap*ucoeff(:,j);
        lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
        
        swim_coeff=MSwim*dxu_coeff(:,j);
        
        DT_coeff=DT*dx2u_coeff(:,j);
        
        adv_comb_coeff(:,j)=adv_coeff+swim_coeff-DT_coeff;

        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_comb_coeff(:,j))+rho(k)*adv_p_coeff(:,j))...
            -Kp/alpha(k)*(int_const-Nint_loc)*Mint'.*ucoeff(:,j); %#ok<*PFBNS>
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
    end
    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dx;
    
    %% RK step 3
    k=3;
    dxu_coeff=ucoeff*Rdx;
    dx2u_coeff=ucoeff*Rd2x;
    adv_p_coeff=adv_comb_coeff;
    parfor j=1:N_mesh
       
        adv_coeff=S_profile(j)*(Mvor*ucoeff(:,j))+Mgyro*ucoeff(:,j);
        adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
        
        lap_coeff=Mlap*ucoeff(:,j);
        lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
        
        swim_coeff=MSwim*dxu_coeff(:,j);
        
        DT_coeff=DT*dx2u_coeff(:,j);
        
        adv_comb_coeff(:,j)=adv_coeff+swim_coeff-DT_coeff;

        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_comb_coeff(:,j))+rho(k)*adv_p_coeff(:,j))...
            -Kp/alpha(k)*(int_const-Nint_loc)*Mint'.*ucoeff(:,j);
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
        
    end
    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dx;
    
    %% Saving for Post-Processing    
    % Saving full Psi and it time derivative
    PS=PS.RunTimeCall(ucoeff,i);
%     if ( mod(i, saving_rate2) == 0 )
%         ufull_save=ucoeff;
%         t=i*dt;
%     end
%     if ( mod(i, saving_rate2) == 2 )&& i~=2 
%         fdt_full_save=((-ucoeff./(real(Mint*ucoeff*2*pi))...
%             + ucoeff_previous2(:,:,1)./(real(Mint*ucoeff_previous2(:,:,1)*2*pi)))/12 ...
%             +(ucoeff_previous2(:,:,3)./(real(Mint*ucoeff_previous2(:,:,3)*2*pi))...
%             -ucoeff_previous2(:,:,2)./(real(Mint*ucoeff_previous2(:,:,2)*2*pi)))*(2/3))/dt;
%         udt_full_save=((-ucoeff...
%             + ucoeff_previous2(:,:,1))/12 ...
%             +(ucoeff_previous2(:,:,3)...
%             -ucoeff_previous2(:,:,2))*(2/3))/dt;
%         save(['t' num2str(t) '.mat'],'t','ufull_save','fdt_full_save','udt_full_save');
%     end
%     if ( mod(i, saving_rate2) == 1 )&& i~=1 
%         ucoeff_previous2(:,:,3)=ucoeff;
%     end
%     if ( mod(i, saving_rate2) == saving_rate2-1 )
%         ucoeff_previous2(:,:,2)=ucoeff;
%     end
%     if ( mod(i, saving_rate2) == saving_rate2-2 )
%         ucoeff_previous2(:,:,1)=ucoeff;
%     end
    
    % Saving Cell Density
    if ( mod(i, saving_rate3) == 0 )
        cell_den(i/saving_rate3,:)=cell_den_loc;
    end    

end
