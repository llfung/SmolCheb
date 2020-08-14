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
Vc=1;                       % Swimming Speed (scaled by channel width and Dr) (Pe_s)
Pef=1;                      % Flow Peclet Number (Pe_f)
% Vsmin=0.2;                  % Minimum sedimentaion (Vs)
Vsvar=0.2;                  % Vs_max-Vs_min

diff_const = 1;             % Rotational Diffusion constant
DT=.0;                      % Translational Diffusion constant
beta=2.2;                   % Gyrotactic time scale
AR=20;                      % Aspect Ratio of swimmer (1=spherical) % AR=1.3778790674938353091971374518539773339097820167847;
B=(AR^2-1)/(AR^2+1);        % Bretherton Constant of swimmer (a.k.a. alpha0)

dt = 0.01;                  % Time step
tfinal = 20+dt*2;           % Stopping time
nsteps = ceil(tfinal/dt);   % Number of time steps
m = 16;                     % Spatial discretization - phi (even)
n = 20;                     % Spaptial discretization - theta (even)
N_mesh=100;                 % Spaptial discretization - x

omg=[0,-1,0];               % Vorticity direction (1,2,3) 

% Run saving settings
saving_rate1=1000;
saving_rate2=1000;
saving_rate3=25;

% x_sav_location=[1 11 21 33 24 3 42 45 48];
x_sav_location=[1 11 26 31 51];

%Saving to settings struct
settings.beta=beta;
settings.n=n;
settings.m=m;
settings.omg1=omg(1);
settings.omg2=omg(2);
settings.omg3=omg(3);
settings.e11=0;
settings.e12=0;
settings.e13=1;
settings.e22=0;
settings.e23=0;
settings.e33=0;

%% x-domain Meshing
dx=2/(N_mesh);
x=-1:dx:1-dx;
% cheb=chebyshev(N_mesh,2,bc_type.none,tran_type.none);
% x=cheb.col_pt;
% D1=(cheb.D(1))';

%% Initial Condition
int_const=1.;
settings.int_const=int_const;

ucoeff0=zeros(n*m,N_mesh);ucoeff0(m*n/2+m/2+1,:)=1/8/pi;

%% Shear Profile
% W_profile=(-cos(pi*x)-1)*Pef;   % W(x)=-cos(pi x)-1
S_profile=pi*sin(pi*x)*Pef/2; % .5*dW(x)/dx=pi*sin(pi x)/2
S_profile(1)=0;

%S_profile=x*Pef; % W(x)=-(1-x^2)
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
Mvor=adv_vor_mat(settings)+B*adv_strain_mat(settings);
Mgyro=settings.beta*adv_gyro_mat(settings);

%Laplacian
Mlap=lap_mat(settings);
helm=helmholtz_gen( n, m);

%Dx
Rdx=spdiags(ones(N_mesh,1)*[-1/60 3/20 -3/4 0 3/4 -3/20 1/60],[3:-1:-3],N_mesh,N_mesh);
Rdx=spdiags(ones(N_mesh,1)*[-1/60 3/20 -3/4 3/4 -3/20 1/60],[-N_mesh+3:-1:-N_mesh+1 N_mesh-1:-1:N_mesh-3],Rdx);
Rdx=Rdx/dx;
Rd2x=spdiags(ones(N_mesh,1)*[1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90],[3:-1:-3],N_mesh,N_mesh);
Rd2x=spdiags(ones(N_mesh,1)*[1/90 -3/20 3/2 3/2 -3/20 1/90],[-N_mesh+3:-1:-N_mesh+1 N_mesh-1:-1:N_mesh-3],Rd2x);
Rd2x=Rd2x/dx/dx;

%p1
Mp1 = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));
%p3
Mp3 = kron(spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n),speye(m));
%p1p3
Mp1p3 = kron(spdiags(.25i*ones(n,1)*[-1,1], [-2 2], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m));

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
Va=NaN(floor(nsteps/saving_rate3),N_mesh);
DDT=NaN(floor(nsteps/saving_rate3),N_mesh);

%% Time-Stepping (RK3-CN2)
ucoeff=ucoeff0;
adv_p_coeff   =zeros(n*m,N_mesh);
adv_comb_coeff=zeros(n*m,N_mesh);
ucoeff_previous=NaN(n*m,N_mesh,3);
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
        
        swim_coeff=(Vc*Mp1-Vsvar*Mp1p3)*dxu_coeff(:,j);
        
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
        
        swim_coeff=(Vc*Mp1-Vsvar*Mp1p3)*dxu_coeff(:,j);  
        
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
        
        swim_coeff=(Vc*Mp1-Vsvar*Mp1p3)*dxu_coeff(:,j);

        DT_coeff=DT*dx2u_coeff(:,j);
        
        adv_comb_coeff(:,j)=adv_coeff+swim_coeff-DT_coeff;

        rhs_coeff = -K2/alpha(k)*ucoeff(:,j)-lap_coeff+1/diff_const/alpha(k)*(gamma(k)*(adv_comb_coeff(:,j))+rho(k)*adv_p_coeff(:,j))...
            -Kp/alpha(k)*(int_const-Nint_loc)*Mint'.*ucoeff(:,j);
        ucoeff(:,j) = helmholtz_cal(rhs_coeff, -K2/alpha(k),helm);
        
    end
    cell_den_loc=real(Mint*ucoeff*2*pi);
    Nint_loc=sum(cell_den_loc,2)*dx;
    
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
        d2xf=f*Rd2x;
        dxf=f*Rdx;
%         d2zf=f*Rd2z;
%         dzf=f*Rdz;
        ex_avg=real(Mint*Mp1*f*(2*pi));
        ez_avg=real(Mint*Mp3*f*(2*pi));
        
        bx_RHS=Mp1*f-ex_avg.*f;
        bz_RHS=Mp3*f-ez_avg.*f;
        inhomo_p1_RHS=Mp1*(dxf)-(ex_avg*Rdx).*f;
        % inhomo_p3_RHS=Mp3*(f*Rdz)-(ez_avg*Rdz).*f;
        
        Dxx_temp=NaN(1,N_mesh);
        Dzx_temp=NaN(1,N_mesh);
        Dxz_temp=NaN(1,N_mesh);
        Dzz_temp=NaN(1,N_mesh);
        Vix_temp=NaN(1,N_mesh);
%         Viz_temp=NaN(1,N_mesh);
        Va_temp=NaN(1,N_mesh);
        DDT_temp=NaN(1,N_mesh);
        parfor j=1:N_mesh            
            Le=S_profile(j)*Mvor+Mgyro-Mlap; % TODO: Singular when beta=0

            bx=[Le;Mint]\[bx_RHS(:,j);0];
            bz=[Le;Mint]\[bz_RHS(:,j);0];
            b_DT_p1=[Le;Mint]\[dxf(:,j);0];
%             b_DT_p3=[Le;Mint]\[dzf(:,j);0];
            f_inhomo_p1=[Le;Mint]\[inhomo_p1_RHS(:,j);0];
%             f_inhomo_p3=[Le;Mint]\[inhomo_p3_RHS(:,j);0];
            f_a=[Le;Mint]\[d2xf(:,j);0];
%             f_a=[Le;Mint]\[d2zf(:,j);0];
            
            Dxx_temp(j)=Mint*(Mp1*bx)*2*pi;
            Dxz_temp(j)=Mint*(Mp1*bz)*2*pi;
            Dzx_temp(j)=Mint*(Mp3*bx)*2*pi;
            Dzz_temp(j)=Mint*(Mp3*bz)*2*pi;
            Vix_temp(j)=Mint*(Mp1*f_inhomo_p1)*2*pi;
%             Viz_temp(j)=Mint*(Mp3*f_inhomo_p3)*2*pi;            
            Va_temp(j)=Mint*(f_a)*2*pi;
            DDT_temp(j)=Mint*(b_DT_p1)*2*pi;
        end
        Dxx(i/saving_rate3,:)=Dxx_temp;
        Dxz(i/saving_rate3,:)=Dxz_temp;
        Dzx(i/saving_rate3,:)=Dzx_temp;
        Dzz(i/saving_rate3,:)=Dzz_temp;
        Vix(i/saving_rate3,:)=Vix_temp;
%         Viz(i/saving_rate3,:)=Viz_temp;
        ex(i/saving_rate3,:)=ex_avg;
        ez(i/saving_rate3,:)=ez_avg;
        
        Va(i/saving_rate3,:)=Va_temp;
        DDT(i/saving_rate3,:)=DDT_temp;

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

Nint=sum(cell_den,2)*dx;
% Nint=NaN(size(t1));
% for i=1:length(t1)
%     Nint(i)=cheb.cheb_int(cell_den(i,:)');
% end

ex_file_name=['smol_pBC_' num2str(beta) 'beta_' num2str(B) 'B_' num2str(Vsvar) 'Vsv_' num2str(Vc) 'Vc_' num2str(DT) 'DT_' num2str(Pef) 'Pef_cospi_cd' num2str(N_mesh) '_m' num2str(m) '_n' num2str(n) '_dt' num2str(dt) '_tf' num2str(tfinal)];
ex_file_name=replace(ex_file_name,'.','-');

save([ex_file_name '.mat'],...
    'n','m','N_mesh','nsteps','S_profile','Vc','Pef','omg','beta','diff_const','DT','B','Vsvar',...
    'dt','tfinal','settings','Kp','x','dx',...
    'saving_rate1','saving_rate2','saving_rate3',...
    't1','t2','t3','Nint','cell_den',...
    'ufull_save','u_xloc_save','x_sav_location','ucoeff','ucoeff0',...
    'Dxx','Dxz','Dzx','Dzz','Vix','Vux','ex','Vuz','ez','DDT','Va',...
    'fdt_full_save','fndt_full_save','-v7.3');
exit

