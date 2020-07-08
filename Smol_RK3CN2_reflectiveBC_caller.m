%% Full Smoluchowski Time Stepping solver for the Fokker-Planck Equation (RK3-CN2)
% Loosely based on Spherefun, this solve uses Double Fourier Sphere (DFS)
% method to tranform the orientational space and time-marched in the DFS 
% space. Time stepping is semi-implicit, with Advection terms in RK3 and 
% Laplacian term in CN2. Implicit matrix inversion is done using the
% Spherefun Helmholtz Solver (only slightly modified to remove
% chebfun-based activiities). 
% 
% pc=parcluster('local');
% pc.JobStorageLocation = strcat(getenv('TMPDIR'),'/para_tmp');
% par=parpool(pc,32);

% parpool(20);
clear all;
%% Setting up
% Parameters
dt = 0.002;                  % Time step
tfinal = 1+dt*2;               % Stopping time
nsteps = ceil(tfinal/dt);   % Number of time steps
m = 16;                     % Spatial discretization - phi (even)
n = 16;                     % Spaptial discretization - theta (even)
N_mesh=51;                 % Spaptial discretization - y
diff_const = 1;             % Diffusion constant
beta=2.2;                   % Gyrotactic time scale
% S=2.5;                      % Shear time scale
Vc=1;                       % Swimming Speed (scaled by channel width and Dr) (Pe_s)
Pef=Vc*2/beta*1.1;
% Vc=1;                       % Swimming Speed (scaled by channel width and Dr) (Pe_s)
% Pef=Vc*2;

omg=[0,-1,0];                % Vorticity direction (1,2,3) 

% Run saving settings
saving_rate1=1000;
saving_rate2=50;
saving_rate3=50;

% x_sav_location=[1 11 21 33 24 3 42 45 48];
x_sav_location=[1 11 26 31 51];
%Saving to settings struct
% settings.S=S;
settings.beta=beta;
settings.n=n;
settings.m=m;
settings.omg1=omg(1);
settings.omg2=omg(2);
settings.omg3=omg(3);


%% x-domain Meshing
% dx=2/(N_mesh);
% x=-1:dx:1-dx;
cheb=chebyshev(N_mesh,2,bc_type.none,tran_type.none);
x=cheb.col_pt;
D1=(cheb.D(1))';

%% Initial Condition
int_const=1.;
settings.int_const=int_const;

ucoeff0=zeros(n*m,N_mesh);ucoeff0(m*n/2+m/2+1,:)=1/8/pi;

%% Shear Profile
% W_profile=(-cos(pi*x)-1)*Pef;   % W(x)=-cos(pi x)-1
% S_profile=pi*sin(pi*x)*Pef/2; % W(x)=-cos(pi x)-1
% S_profile(1)=0;

S_profile=x*Pef; % W(x)=-(1-x^2)
% S_profile=Pef/2*ones(size(x)); % W(x)=x
%% RK3 coeff and constants
alpha=[4/15 1/15 1/6];
gamma=[8/15 5/12 3/4];
rho=[0 -17/60 -5/12];

% Preparing Constants
K2 = (1/(dt*diff_const));         % Helmholtz frequency for BDF1

%% Initialising Matrices
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
Kp=1;
settings.Kp=Kp/settings.MintSq/diff_const/dt;
Kp=settings.Kp;

% Advection
% Madv=adv_mat(settings);
Mvor=adv_vor_mat(settings);
Mgyro=settings.beta*adv_gyro_mat(settings);

%Laplacian
Mlap=lap_mat(settings);
helm=helmholtz_gen( n, m);

%Dx
Rdx=D1;

%p1
Mp1 = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));
%p3
Mp3 = kron(spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n),speye(m)); %e3

%% Initialise Recorded values
cell_den=NaN(floor(nsteps/saving_rate3),N_mesh);

ufull_save=NaN(n*m,N_mesh,floor(nsteps/saving_rate2));
fdt_full_save=NaN(n*m,N_mesh,floor(nsteps/saving_rate2));
fndt_full_save=NaN(n*m,N_mesh,floor(nsteps/saving_rate2));

u_xloc_save=NaN(n*m,floor(nsteps/saving_rate1),length(x_sav_location));

Dxx=NaN(floor(nsteps/saving_rate3),N_mesh);
Dzx=NaN(floor(nsteps/saving_rate3),N_mesh);
Dxz=NaN(floor(nsteps/saving_rate3),N_mesh);
Dzz=NaN(floor(nsteps/saving_rate3),N_mesh);
Vix=NaN(floor(nsteps/saving_rate3),N_mesh);
% Viz=NaN(floor(nsteps/saving_rate3),N_mesh);
Vux=NaN(floor(nsteps/saving_rate3),N_mesh);
Vuz=NaN(floor(nsteps/saving_rate3),N_mesh);
ex=NaN(floor(nsteps/saving_rate3),N_mesh);
ez=NaN(floor(nsteps/saving_rate3),N_mesh);

%% Time-Stepping (RK3-CN2)
ucoeff=ucoeff0;
adv_p_coeff   =zeros(n*m,N_mesh);
adv_comb_coeff=zeros(n*m,N_mesh);
ucoeff_previous=NaN(n*m,N_mesh,3);
ucoeff_previous2=NaN(n*m,N_mesh,3);

    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=cheb.cheb_int(cell_den_loc');
    
for i = 1:nsteps
    %% RK step 1
    k=1;
    % Par-For Version
    dxu_coeff=ucoeff*Rdx;
    parfor j=1:N_mesh
        adv_coeff=S_profile(j)*(Mvor*ucoeff(:,j))+Mgyro*ucoeff(:,j);
        adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
        
        lap_coeff=Mlap*ucoeff(:,j);
        lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
        
        swim_coeff=Vc*Mp1*dxu_coeff(:,j);
        
        adv_p_coeff(:,j)=adv_coeff+swim_coeff;
        if j==1 || j==N_mesh
            adv_col=transpose(reshape(adv_p_coeff(:,j),m,n));
            floorm=floor(m/2);
            if mod(m,2)
            for l=1:floorm
                adv_col(:,floorm+1-l)=(adv_col(:,floorm+1-l)+(-1)^l*adv_col(:,floorm+1+l))/2;
                adv_col(:,floorm+1+l)=(-1)^l*adv_col(:,floorm+1-l);
            end
            else
                adv_col(:,1)=0;
            for l=1:floorm-1
                adv_col(:,floorm+1-l)=(adv_col(:,floorm+1-l)+(-1)^l*adv_col(:,floorm+1+l))/2;
                adv_col(:,floorm+1+l)=(-1)^l*adv_col(:,floorm+1-l);
            end       
            end
            adv_p_coeff(:,j)=reshape(transpose(adv_col),n*m,1);
        end
        
        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_p_coeff(:,j)))...
            -Kp/alpha(k)*(int_const-Nint_loc)*Mint'.*ucoeff(:,j);
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
    end
    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=cheb.cheb_int(cell_den_loc');
    
    %% RK step 2
    k=2;
    dxu_coeff=ucoeff*Rdx;
    parfor j=1:N_mesh
%         adv_p_coeff=adv_coeff+swim_coeff;
        adv_coeff=S_profile(j)*(Mvor*ucoeff(:,j))+Mgyro*ucoeff(:,j);
        adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
        
        lap_coeff=Mlap*ucoeff(:,j);
        lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
        
        swim_coeff=Vc*Mp1*dxu_coeff(:,j);
        adv_comb_coeff(:,j)=adv_coeff+swim_coeff;
        if j==1 || j==N_mesh
            adv_col=transpose(reshape(adv_comb_coeff(:,j),m,n));
            floorm=floor(m/2);
            if mod(m,2)
            for l=1:floorm
                adv_col(:,floorm+1-l)=(adv_col(:,floorm+1-l)+(-1)^l*adv_col(:,floorm+1+l))/2;
                adv_col(:,floorm+1+l)=(-1)^l*adv_col(:,floorm+1-l);
            end
            else
                adv_col(:,1)=0;
            for l=1:floorm-1
                adv_col(:,floorm+1-l)=(adv_col(:,floorm+1-l)+(-1)^l*adv_col(:,floorm+1+l))/2;
                adv_col(:,floorm+1+l)=(-1)^l*adv_col(:,floorm+1-l);
            end       
            end
            adv_comb_coeff(:,j)=reshape(transpose(adv_col),n*m,1);
        end
        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_comb_coeff(:,j))+rho(k)*adv_p_coeff(:,j))...
            -Kp/alpha(k)*(int_const-Nint_loc)*Mint'.*ucoeff(:,j); %#ok<*PFBNS>
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
    end
    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=cheb.cheb_int(cell_den_loc');
    
    %% RK step 3
    k=3;
    dxu_coeff=ucoeff*Rdx;
    adv_p_coeff=adv_comb_coeff;
    parfor j=1:N_mesh
       
        adv_coeff=S_profile(j)*(Mvor*ucoeff(:,j))+Mgyro*ucoeff(:,j);
        adv_coeff=adv_coeff-Mint'*(Mint*adv_coeff)/MintSq;
        
        lap_coeff=Mlap*ucoeff(:,j);
        lap_coeff=lap_coeff-Mint'*(Mint*lap_coeff)/MintSq;
        
        swim_coeff=Vc*Mp1*dxu_coeff(:,j);
        adv_comb_coeff(:,j)=adv_coeff+swim_coeff;
        if j==1 || j==N_mesh
            adv_col=transpose(reshape(adv_comb_coeff(:,j),m,n));
            floorm=floor(m/2);
            if mod(m,2)
            for l=1:floorm
                adv_col(:,floorm+1-l)=(adv_col(:,floorm+1-l)+(-1)^l*adv_col(:,floorm+1+l))/2;
                adv_col(:,floorm+1+l)=(-1)^l*adv_col(:,floorm+1-l);
            end
            else
                adv_col(:,1)=0;
            for l=1:floorm-1
                adv_col(:,floorm+1-l)=(adv_col(:,floorm+1-l)+(-1)^l*adv_col(:,floorm+1+l))/2;
                adv_col(:,floorm+1+l)=(-1)^l*adv_col(:,floorm+1-l);
            end       
            end
            adv_comb_coeff(:,j)=reshape(transpose(adv_col),n*m,1);
        end
        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_comb_coeff(:,j))+rho(k)*adv_p_coeff(:,j))...
            -Kp/alpha(k)*(int_const-Nint_loc)*Mint'.*ucoeff(:,j);
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
        
    end
    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=cheb.cheb_int(cell_den_loc');
    
    %% Saving for Post-Processing
    if ( mod(i, saving_rate1) == 0 )
        for j=1:length(x_sav_location)
            u_xloc_save(:,i/saving_rate1,j)=ucoeff(:,x_sav_location(j));
        end
    end
    
    %    Plot/Save the solution every saving_rate
    if ( mod(i, saving_rate2) == 0 )
        ufull_save(:,:,i/saving_rate2)=ucoeff;
    end
    if ( mod(i, saving_rate2) == 2 )&& i~=2 
        fdt_full_save(:,:,(i-2)/saving_rate2)=((-ucoeff./(real(Mint*ucoeff*2*pi))...
            + ucoeff_previous2(:,:,1)./(real(Mint*ucoeff_previous2(:,:,1)*2*pi)))/12 ...
            +(ucoeff_previous2(:,:,3)./(real(Mint*ucoeff_previous2(:,:,3)*2*pi))...
            -ucoeff_previous2(:,:,2)./(real(Mint*ucoeff_previous2(:,:,2)*2*pi)))*(2/3))/dt;
        fndt_full_save(:,:,(i-2)/saving_rate2)=((-ucoeff...
            + ucoeff_previous2(:,:,1))/12 ...
            +(ucoeff_previous2(:,:,3)...
            -ucoeff_previous2(:,:,2))*(2/3))/dt;
    end
    if ( mod(i, saving_rate2) == 1 )&& i~=1 
        ucoeff_previous2(:,:,3)=ucoeff;
    end
    if ( mod(i, saving_rate2) == saving_rate2-1 )
        ucoeff_previous2(:,:,2)=ucoeff;
    end
    if ( mod(i, saving_rate2) == saving_rate2-2 )
        ucoeff_previous2(:,:,1)=ucoeff;
    end
    
    %% On-the-go-Post-Processing
     if ( mod(i, saving_rate3) == 0 )
        cellden_temp=real(Mint*ucoeff*2*pi);
        f=ucoeff./cellden_temp;
        ex_avg=real(Mint*Mp1*f*(2*pi));
        ez_avg=real(Mint*Mp3*f*(2*pi));
        
        bx_RHS=Mp1*f-ex_avg.*f;
        bz_RHS=Mp3*f-ez_avg.*f;
        inhomo_p1_RHS=Mp1*(f*Rdx)-(ex_avg*Rdx).*f;
        % inhomo_p3_RHS=Mp3*(f*Rdz)-(ez_avg*Rdz).*f;
        
        Dxx_temp=NaN(1,N_mesh);
        Dzx_temp=NaN(1,N_mesh);
        Dxz_temp=NaN(1,N_mesh);
        Dzz_temp=NaN(1,N_mesh);
        Vix_temp=NaN(1,N_mesh);
%         Viz_temp=NaN(1,N_mesh);
        for j=1:N_mesh            
            Le=S_profile(j)*Mvor+Mgyro-Mlap; % TODO: Singular when beta=0

            bx=[Le;Mint]\[bx_RHS(:,j);0];
            bz=[Le;Mint]\[bz_RHS(:,j);0];
            f_inhomo_p1=[Le;Mint]\[inhomo_p1_RHS(:,j);0];
%             f_inhomo_p3=[Le;Mint]\[inhomo_p3_RHS(:,j);0];
            
            Dxx_temp(j)=Mint*(Mp1*bx)*2*pi;
            Dxz_temp(j)=Mint*(Mp1*bz)*2*pi;
            Dzx_temp(j)=Mint*(Mp3*bx)*2*pi;
            Dzz_temp(j)=Mint*(Mp3*bz)*2*pi;
            Vix_temp(j)=Mint*(Mp1*f_inhomo_p1)*2*pi;
%             Viz_temp(j)=Mint*(Mp3*f_inhomo_p3)*2*pi;
        end
        Dxx(i/saving_rate3,:)=Dxx_temp;
        Dxz(i/saving_rate3,:)=Dxz_temp;
        Dzx(i/saving_rate3,:)=Dzx_temp;
        Dzz(i/saving_rate3,:)=Dzz_temp;
        Vix(i/saving_rate3,:)=Vix_temp;
%         Viz(i/saving_rate3,:)=Viz_temp;
        ex(i/saving_rate3,:)=ex_avg;
        ez(i/saving_rate3,:)=ez_avg;
        cell_den(i/saving_rate3,:)=cellden_temp;
        
        disp([num2str(i) '/' num2str(nsteps)]);
    end 
    if ( mod(i, saving_rate3) == (saving_rate3-2) )
        ucoeff_previous(:,:,1)=ucoeff;
    end 
    if ( mod(i, saving_rate3) == (saving_rate3-1) )
        ucoeff_previous(:,:,2)=ucoeff;
    end 
    if ( mod(i, saving_rate3) == 1 ) && i~=1
        ucoeff_previous(:,:,3)=ucoeff;
    end 
    if ( mod(i, saving_rate3) == 2 ) && i~=2
        unsteady_RHS=((-ucoeff./(real(Mint*ucoeff*2*pi))...
            + ucoeff_previous(:,:,1)./(real(Mint*ucoeff_previous(:,:,1)*2*pi)))/12 ...
            +(ucoeff_previous(:,:,3)./(real(Mint*ucoeff_previous(:,:,3)*2*pi))...
            -ucoeff_previous(:,:,2)./(real(Mint*ucoeff_previous(:,:,2)*2*pi)))*(2/3))/dt;
        Vux_temp=NaN(1,N_mesh);
        Vuz_temp=NaN(1,N_mesh);
        parfor j=1:N_mesh
            Le=S_profile(j)*Mvor+Mgyro-Mlap; % TODO: Singular when beta=0

            f_unsteady=[Le;Mint]\[unsteady_RHS(:,j);0];
            
            Vux_temp(j)=Mint*(Mp1*f_unsteady)*2*pi;
            Vuz_temp(j)=Mint*(Mp3*f_unsteady)*2*pi;
        end
        Vux((i-2)/saving_rate3,:)=Vux_temp;
        Vuz((i-2)/saving_rate3,:)=Vuz_temp;
    end 
end


%% Surface Integral Conservation check
t1=dt*saving_rate1:dt*saving_rate1:tfinal;
t2=dt*saving_rate2:dt*saving_rate2:tfinal;
t3=dt*saving_rate3:dt*saving_rate3:tfinal;

% Nint=sum(cell_den,2)*dx;
Nint=NaN(size(t3));
for i=1:length(t3)
    Nint(i)=cheb.cheb_int(cell_den(i,:)');
end

ex_file_name=['smol_rBC_' num2str(beta) 'beta_' num2str(Vc) 'Vc_' num2str(Pef) 'Pef_para_cheb' num2str(N_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];
ex_file_name=replace(ex_file_name,'.','-');

save([ex_file_name '_wPS.mat'],...
    'n','m','N_mesh','nsteps','S_profile','Vc','Pef','omg','beta','diff_const',...
    'dt','tfinal','settings','x',...
    'cheb',... 'dx'
    'saving_rate1','saving_rate2','saving_rate3',...
    't1','t2','t3','Nint','cell_den',...
    'ufull_save','u_xloc_save','x_sav_location','ucoeff','ucoeff0',...
    'Dxx','Dxz','Dzx','Dzz','Vix','Vux','ex','Vuz','ez',... 'Viz',
    'fdt_full_save','fndt_full_save','-v7.3');
exit
