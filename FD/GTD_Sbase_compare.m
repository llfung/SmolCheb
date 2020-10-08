%% GTD Library Generation 
% Using coordinate change (omega_e1=0), normalisation (|omega|=1) and
% diagonalisation (purely imaginary), generate library that can be
% interpolated and recreat diffusivity and average swimming. 
clear all
% par=parpool(28);
%% Define discretization parameters
settings.n_phi=64; % Has to be even for FFT. 2^N recommended
settings.n_theta=201; % Had better to be 1+(multiples of 5,4,3 or 2) for Newton-Cote
settings.tn_phi=settings.n_phi/2+1;
settings.tn_theta=(floor(settings.n_theta/2)+1);
settings.N_total=(settings.n_theta)*settings.n_phi;
settings.dtheta=(pi/(settings.n_theta-1));
settings.dphi=2*pi/(settings.n_phi);
% N_total=settings.n_phi*settings.n_theta;
settings.theta=[0:settings.dtheta:pi]';
settings.phi=[0:settings.dphi:(2*pi-settings.dphi)];
settings.kphi=[0:(settings.tn_phi-1)];

e1_field=sin(settings.theta)*cos(settings.phi);
e2_field=sin(settings.theta)*sin(settings.phi);
e3_field=cos(settings.theta)*ones(size(settings.phi));


e_all_field(:,:,1)=e1_field;
e_all_field(:,:,2)=e2_field;
e_all_field(:,:,3)=e3_field;
b_endings=[1 settings.n_theta settings.n_theta+1 2*settings.n_theta 2*settings.n_theta+1 3*settings.n_theta];

%% Newton-Cote Integrand
    if mod(settings.n_theta,5)==1
        settings.integrad=[0 sin(settings.theta(2:end-1)') 0].*[19 repmat([75 50 50 75 38],1,(settings.n_theta-6)/5)  75 50 50 75 19]*5/288*settings.dtheta*2*pi;
    elseif mod(settings.n_theta,4)==1
        settings.integrad=[0 sin(settings.theta(2:end-1)') 0].*[7 repmat([32 12 32 14],1,(settings.n_theta-5)/4)  32 12 32 7]*2/45*settings.dtheta*2*pi;
    elseif mod(settings.n_theta,3)==1
        settings.integrad=[0 sin(settings.theta(2:end-1)') 0].*[1 repmat([3 3 2],1,(settings.n_theta-4)/3) 3 3 1]*3/8*settings.dtheta*2*pi;
    elseif mod(settings.n_theta,2)==1
        settings.integrad=[0 sin(settings.theta(2:end-1)') 0].*[1 repmat([4 2],1,(N-3)/2) 4 1]/3*settings.dtheta*2*pi;
    else
        settings.integrad=[0 sin(settings.theta(2:end-1)') 0]*settings.dtheta*pi*2; %% Trapezoid Rule
    end

settings.int=kron(settings.integrad,ones(1,settings.n_phi)/settings.n_phi);
%% Define input parameters
settings.beta=2.2;

% Preparing Struct
G11=0;
G12=0;
G13=1;
G21=0;
G22=0;
G23=0;
G31=0;
G32=0;
G33=0;
G=[G11 G21 G31;G12 G22 G32;G13 G23 G33];

settings.omega1=G23-G32;
settings.omega3=G12-G21;
settings.omega2=G31-G13;

settings.E11=G11;
settings.E12=(G12+G21)/2;
settings.E13=(G13+G31)/2;
settings.E22=G22;
settings.E23=(G23+G32)/2;
settings.E33=G33;

settings.B=0;
%% Initialisation
S_loop=[-30:0.1:30];
% G13_loop=[-exp(6:-0.025:-12) 0 exp(-12:0.025:6)];

% Looping Mesh
N_loop=numel(S_loop);

% Initialise result array
res_array=zeros(N_loop,9);
rese_array=zeros(N_loop,9);
resG_array=zeros(N_loop,6);
bf_array=zeros(N_loop,3);
e1_array=zeros(N_loop,1);
e2_array=zeros(N_loop,1);
e3_array=zeros(N_loop,1);


%% Check Arrays
%     eig_check_array=zeros(size(res_array));
%     S_check=zeros(size(res_array));
%     omega3_check=zeros(size(res_array));
%% Loop for different S
parfor ii=1:N_loop
    %% Set up parameters in loop
    settings_loc=settings;
%     settings_loc.S=S_mesh(ii);
    settings_loc.omega1=settings_loc.omega1*S_loop(ii);
    settings_loc.omega2=settings_loc.omega2*S_loop(ii);
    settings_loc.omega3=settings_loc.omega3*S_loop(ii);

    settings_loc.E11=settings_loc.E11*S_loop(ii);
    settings_loc.E12=settings_loc.E12*S_loop(ii);
    settings_loc.E13=settings_loc.E13*S_loop(ii);
    settings_loc.E22=settings_loc.E22*S_loop(ii);
    settings_loc.E23=settings_loc.E23*S_loop(ii);
    settings_loc.E33=settings_loc.E33*S_loop(ii);
    
    %% Solving for f(e)
    LHSL=L_FD_LHS(settings_loc);
    LHS_F=L_FD_LHS_BC(LHSL,settings_loc);
%     LHS_F_=LHS_F;
    [V,~]=eigs(LHS_F,1,0);
%     [Usvd,Ssvd,Vsvd]=svds(LHS_F,1,'smallest');

%     LHS_Flang=[LHS_F settings_loc.int';settings_loc.int 0];
%     RHS_lang=[zeros(settings_loc.N_total,1);1];
%     ans_lang=LHS_lang\RHS_lang;
%     f0_lang=transpose(reshape(ans_lang(1:settings_loc.N_total,1),settings_loc.n_phi,settings_loc.n_theta));
    
%     LHSmod=LHS_F_-Usvd*Ssvd*Vsvd'+int'*Vsvd'*sqrt(int*int');
    f_sol=V/(kron(settings_loc.integrad,ones(1,settings_loc.n_phi)/settings_loc.n_phi)*V);
%     LHS_F(settings_loc.N_total/2,:)=kron(settings_loc.integrad,ones(1,settings_loc.n_phi)/settings_loc.n_phi);
%     RHS_F=zeros(settings_loc.n_theta*settings_loc.n_phi,1);
%     RHS_F(settings_loc.N_total/2,1)=1;
%     f_sol=LHS_F\RHS_F;
    f0=transpose(reshape(f_sol,settings_loc.n_phi,settings_loc.n_theta));

    %% Finding e_avg 
    ebracket=eavg(f0,e_all_field,settings_loc.integrad);  
    % In (1,2,3) local Coordinate
    e1_array(ii)=ebracket.e1;
    e2_array(ii)=ebracket.e2;
    e3_array(ii)=ebracket.e3;    

     %% Solving for b(e)
     % LHS
    bG=kron(G*S_loop(ii),speye(settings_loc.n_theta*settings_loc.n_phi));
    
    LHS=-kron(eye(3),LHSL)-bG;
    
    % Applying Boundary Condition for LHS Matrix for solving b(e)
    %   By assuming df\dtheta=0 at theta=0,pi, in 6th order,
    %   using average over phi 
    BC1=kron(sparse([-49/20 zeros(1,settings_loc.n_theta-1);...
        zeros(settings_loc.n_theta-2,settings_loc.n_theta);...
        zeros(1,settings_loc.n_theta-1) 49/20]),speye(settings_loc.n_phi))/settings_loc.dtheta;
    BC2=kron(sparse([0 6 -15/2 20/3 -15/4 6/5 -1/6 zeros(1,settings_loc.n_theta-7);...
        zeros(settings_loc.n_theta-2,settings_loc.n_theta);...
        zeros(1,settings_loc.n_theta-7) 1/6 -6/5 15/4 -20/3 15/2 -6 0]),sparse(ones(settings_loc.n_phi,settings_loc.n_phi)))/settings_loc.dtheta/settings_loc.n_phi;

    LHS(                                          1:                       settings_loc.n_phi,:)=0;
    LHS(  settings_loc.N_total-settings_loc.n_phi+1:  settings_loc.N_total,                   :)=0;
    LHS(  settings_loc.N_total                   +1:  settings_loc.N_total+settings_loc.n_phi,:)=0;
    LHS(2*settings_loc.N_total-settings_loc.n_phi+1:2*settings_loc.N_total,                   :)=0;
    LHS(2*settings_loc.N_total                   +1:2*settings_loc.N_total+settings_loc.n_phi,:)=0;
    LHS(3*settings_loc.N_total-settings_loc.n_phi+1:3*settings_loc.N_total,                   :)=0; 
    
    LHS=LHS+kron(speye(3,3),BC1+BC2);
    
    % RHS
%     f0_col=reshape(transpose(f0),(settings_loc.n_theta)*settings_loc.n_phi,1);
    RHS_mat=[f0.*e_all_field(:,:,1)-f0*ebracket.e1;f0.*e_all_field(:,:,2)-f0*ebracket.e2;f0.*e_all_field(:,:,3)-f0*ebracket.e3];
    RHS_mat(b_endings(1:6),:)=0;
    RHS=reshape(transpose(RHS_mat),3*(settings_loc.n_theta)*settings_loc.n_phi,1);

    % Normalisation (Legacy)
        % Psuedo Inverse
        % LHS=[LHS; kron(speye(3),kron(settings_loc.integrad,ones(1,settings_loc.n_phi)/settings_loc.n_phi))];
        % RHS=[RHS;0;0;0];
        %   Sq. Matrix
        %     LHS(settings_loc.N_total/2  ,:)=[kron(settings_loc.integrad,ones(1,settings_loc.n_phi)/settings_loc.n_phi) zeros(1,2*settings_loc.N_total)];
        %     LHS(settings_loc.N_total/2*3,:)=[zeros(1,  settings_loc.N_total) kron(settings_loc.integrad,ones(1,settings_loc.n_phi)/settings_loc.n_phi) zeros(1,settings_loc.N_total)];
        %     LHS(settings_loc.N_total/2*5,:)=[zeros(1,2*settings_loc.N_total) kron(settings_loc.integrad,ones(1,settings_loc.n_phi)/settings_loc.n_phi)];
        %     RHS([settings_loc.N_total/2 settings_loc.N_total/2*3 settings_loc.N_total/2*5],1)=0;

    % Normalisation (Langrange Multiplier)
    LHS_lang=[LHS [settings_loc.int';zeros(settings_loc.N_total*2,1)] [zeros(settings_loc.N_total,1);settings_loc.int';zeros(settings_loc.N_total,1)] [zeros(settings_loc.N_total*2,1);settings_loc.int'];...
        [[settings_loc.int zeros(1,settings_loc.N_total*2)];[zeros(1,settings_loc.N_total) settings_loc.int zeros(1,settings_loc.N_total)];[zeros(1,settings_loc.N_total*2) settings_loc.int]],zeros(3,3)];
    RHS_lang=[RHS;zeros(3,1)];
    
    
    % Solving
    b_col_lang=LHS_lang\RHS_lang;
    b=transpose(reshape(b_col_lang(1:settings_loc.N_total*3),settings_loc.n_phi,settings_loc.n_theta*3));
    
    b1=b(b_endings(1):b_endings(2),:);%b1_col=reshape(transpose(b1),(settings_loc.n_theta)*settings_loc.n_phi,1);
    b2=b(b_endings(3):b_endings(4),:);%b2_col=reshape(transpose(b2),(settings_loc.n_theta)*settings_loc.n_phi,1);
    b3=b(b_endings(5):b_endings(6),:);%b3_col=reshape(transpose(b3),(settings_loc.n_theta)*settings_loc.n_phi,1);
        
    %% Finding Diffusitivity
    b1G=(b1*G11+b2*G21+b3*G31)*S_loop(ii);
    b2G=(b1*G12+b2*G22+b3*G32)*S_loop(ii);
    b3G=(b1*G13+b2*G23+b3*G33)*S_loop(ii);

    D11t=b1.*e_all_field(:,:,1)+b1.*b1G./f0; %#ok<*PFBNS>
    D12t=b1.*e_all_field(:,:,2)+b1.*b2G./f0;
    D13t=b1.*e_all_field(:,:,3)+b1.*b3G./f0;
    D21t=b2.*e_all_field(:,:,1)+b2.*b1G./f0;
    D22t=b2.*e_all_field(:,:,2)+b2.*b2G./f0;
    D23t=b2.*e_all_field(:,:,3)+b2.*b3G./f0;
    D31t=b3.*e_all_field(:,:,1)+b3.*b1G./f0;
    D32t=b3.*e_all_field(:,:,2)+b3.*b2G./f0;
    D33t=b3.*e_all_field(:,:,3)+b3.*b3G./f0;
    
    res_temp=zeros(9,1);
    res_temp(1)=settings_loc.integrad*mean(D11t,2);
    res_temp(2)=settings_loc.integrad*mean(D12t,2);
    res_temp(3)=settings_loc.integrad*mean(D13t,2);
    res_temp(4)=settings_loc.integrad*mean(D21t,2);
    res_temp(5)=settings_loc.integrad*mean(D22t,2);
    res_temp(6)=settings_loc.integrad*mean(D23t,2);
    res_temp(7)=settings_loc.integrad*mean(D31t,2);
    res_temp(8)=settings_loc.integrad*mean(D32t,2);
    res_temp(9)=settings_loc.integrad*mean(D33t,2);
    res_array(ii,:)=res_temp;
    %     %Symmetric Part
    D12t=(D12t+D21t)/2;
    D23t=(D23t+D32t)/2;
    D13t=(D13t+D31t)/2;
    res_temp_GTD=zeros(6,1);
    res_temp_GTD(1)=settings_loc.integrad*mean(D11t,2);
    res_temp_GTD(2)=settings_loc.integrad*mean(D12t,2);
    res_temp_GTD(3)=settings_loc.integrad*mean(D22t,2);
    res_temp_GTD(4)=settings_loc.integrad*mean(D13t,2);
    res_temp_GTD(5)=settings_loc.integrad*mean(D23t,2);
    res_temp_GTD(6)=settings_loc.integrad*mean(D33t,2);    
    resG_array(ii,:)=res_temp_GTD;
    
    %% b Euler
    RHS_b1_euler=f0.*(e_all_field(:,:,1)-ebracket.e1);RHS_b1_euler(b_endings(1:2),:)=0;
    RHS_b2_euler=f0.*(e_all_field(:,:,2)-ebracket.e2);RHS_b2_euler(b_endings(1:2),:)=0;
    RHS_b3_euler=f0.*(e_all_field(:,:,3)-ebracket.e3);RHS_b3_euler(b_endings(1:2),:)=0;
    
    LHS_euler=[LHS_F settings_loc.int';settings_loc.int 0];

    RHS_b1_euler_col=reshape(transpose(RHS_b1_euler),(settings_loc.n_theta)*settings_loc.n_phi,1);
    RHS_b2_euler_col=reshape(transpose(RHS_b2_euler),(settings_loc.n_theta)*settings_loc.n_phi,1);
    RHS_b3_euler_col=reshape(transpose(RHS_b3_euler),(settings_loc.n_theta)*settings_loc.n_phi,1);
    
    b1col=(-LHS_euler)\[RHS_b1_euler_col;0];
    b2col=(-LHS_euler)\[RHS_b2_euler_col;0];
    b3col=(-LHS_euler)\[RHS_b3_euler_col;0];
    
    D11e=transpose(reshape(b1col(1:settings_loc.N_total),settings_loc.n_phi,settings_loc.n_theta)).*e_all_field(:,:,1);
    D12e=transpose(reshape(b2col(1:settings_loc.N_total),settings_loc.n_phi,settings_loc.n_theta)).*e_all_field(:,:,1);
    D13e=transpose(reshape(b3col(1:settings_loc.N_total),settings_loc.n_phi,settings_loc.n_theta)).*e_all_field(:,:,1);
    D21e=transpose(reshape(b1col(1:settings_loc.N_total),settings_loc.n_phi,settings_loc.n_theta)).*e_all_field(:,:,2);
    D22e=transpose(reshape(b2col(1:settings_loc.N_total),settings_loc.n_phi,settings_loc.n_theta)).*e_all_field(:,:,2);
    D23e=transpose(reshape(b3col(1:settings_loc.N_total),settings_loc.n_phi,settings_loc.n_theta)).*e_all_field(:,:,2);
    D31e=transpose(reshape(b1col(1:settings_loc.N_total),settings_loc.n_phi,settings_loc.n_theta)).*e_all_field(:,:,3);
    D32e=transpose(reshape(b2col(1:settings_loc.N_total),settings_loc.n_phi,settings_loc.n_theta)).*e_all_field(:,:,3);
    D33e=transpose(reshape(b3col(1:settings_loc.N_total),settings_loc.n_phi,settings_loc.n_theta)).*e_all_field(:,:,3);
    
    rese_temp=zeros(9,1);
    rese_temp(1)=settings_loc.integrad*mean(D11e,2);
    rese_temp(2)=settings_loc.integrad*mean(D12e,2);
    rese_temp(3)=settings_loc.integrad*mean(D13e,2);
    rese_temp(4)=settings_loc.integrad*mean(D21e,2);
    rese_temp(5)=settings_loc.integrad*mean(D22e,2);
    rese_temp(6)=settings_loc.integrad*mean(D23e,2);
    rese_temp(7)=settings_loc.integrad*mean(D31e,2);
    rese_temp(8)=settings_loc.integrad*mean(D32e,2);
    rese_temp(9)=settings_loc.integrad*mean(D33e,2);
    
    rese_array(ii,:)=rese_temp;
    
    disp([num2str(ii) '/' num2str(N_loop)]);
    
end

% res_array=reshape(res_array,numel(G12_loop),numel(G21_loop),numel(G11_loop),numel(G22_loop),NBvar); %#ok<*NASGU>
% e1_array=reshape(e1_array,numel(G12_loop),numel(G21_loop),numel(G11_loop),numel(G22_loop));
% e2_array=reshape(e2_array,numel(G12_loop),numel(G21_loop),numel(G11_loop),numel(G22_loop));
% e3_array=reshape(e3_array,numel(G12_loop),numel(G21_loop),numel(G11_loop),numel(G22_loop));

%% Saving
name=['GTD_beta_' num2str(settings.beta) 'B_' num2str(settings.B) '_GTD_lib_Sbase_Comp'];
clearvars settings e_all_field e1_field e2_field e3_field par;
save([name '.mat']);
% exit