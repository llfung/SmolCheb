function yp=orient_op(~,y,settings)

    %% Initialization
    zi=sqrt(-1);
    
    %% Phi space FFT calculation
    f=transpose(reshape(y,settings.n_phi,settings.n_theta-2));
    
    %% 2nd order BC implementation
    f_theta_0=ones(1,settings.n_phi)*mean(4*f(1,:)-f(2,:))/3;
    f_theta_n=ones(1,settings.n_phi)*mean(4*f(end,:)-f(end-1,:))/3;
    f=[f_theta_0;f;f_theta_n];
    
    cf=forward_fft(f);
    
    %% Phi space FFT calculation
    
    Sf=area(cf,settings.theta);
   % cf(:,1)=cf(:,1)/Sf;
    cdf=zi*((ones(settings.n_theta,1)*settings.kphi).*cf);
    cd2f=-((ones(settings.n_theta,1)*settings.kphi.^2).*cf);
    
    if mod(settings.n_phi,2)==0
        cdf(:,settings.tn_phi)=zeros(settings.n_theta,1);
    end
    df=backward_fft(cdf);
    d2f=backward_fft(cd2f);
    f=backward_fft(cf);
    
    %% RHS Terms
    
    MM=transpose(spdiags([-sin(settings.theta)/(2*settings.dtheta)...
        -2*cos(settings.theta) sin(settings.theta)/(2*settings.dtheta)],...
        -1:1,settings.n_theta,settings.n_theta));

    MM=[zeros(1,settings.n_theta);...
        MM(2:(settings.n_theta-1),:);...
        zeros(1,settings.n_theta)];
    gg=settings.beta*MM*f;

    cot_term=[0;cot(settings.theta(2:(settings.n_theta-1)));0];
    omg=settings.S/2*(cot_term.*df.*(settings.omega_e1*cos(settings.phi)+...
        settings.omega_e3*sin(settings.phi))-settings.omega_e2*df+...
        transpose(spdiags(...
        [ones(settings.n_theta,1)...
        zeros(settings.n_theta,1)...
        -ones(settings.n_theta,1)],...
        -1:1,settings.n_theta,settings.n_theta))*f.*...
        (settings.omega_e1*sin(settings.phi)...
        -settings.omega_e3*cos(settings.phi))/(2*settings.dtheta));

    rx=gg+omg;
    
    pre_ll=transpose(...
        spdiags(...
        [cot(settings.theta)/...
        (2*settings.dtheta)+settings.dtheta^(-2) ...
        -2*settings.dtheta^(-2)*ones(settings.n_theta,1) ...
        -cot(settings.theta)/(2*settings.dtheta)+settings.dtheta^(-2)]...
        ,-1:1,settings.n_theta,settings.n_theta));
    pre_ll=[zeros(1,settings.n_theta);...
        pre_ll(2:(settings.n_theta-1),:);...
        zeros(1,settings.n_theta)];
    ll=pre_ll*f...
        +([0;sin(settings.theta(2:(settings.n_theta-1))).^(-2);0]).*d2f;

    rhs=ll-rx;
    
    %% For checking with L_FD_LHS
%     [LHS,ll_col,rx_col,MM_col,G_col]=L_FD_LHS(settings);
%     rhs_mat=LHS*reshape(transpose(f),(settings.n_theta)*settings.n_phi,1);
%     rhs_comp=transpose(reshape(rhs_mat,settings.n_phi,settings.n_theta));
%     ll_mat=ll_col*reshape(transpose(f),(settings.n_theta)*settings.n_phi,1);
%     ll_comp=transpose(reshape(ll_mat,settings.n_phi,settings.n_theta));
%     rx_mat=rx_col*reshape(transpose(f),(settings.n_theta)*settings.n_phi,1);
%     rx_comp=transpose(reshape(rx_mat,settings.n_phi,settings.n_theta));
%         MM_mat=MM_col*reshape(transpose(f),(settings.n_theta)*settings.n_phi,1);
%     MM_comp=transpose(reshape(MM_mat,settings.n_phi,settings.n_theta));
%         G_mat=G_col*reshape(transpose(f),(settings.n_theta)*settings.n_phi,1);
%     G_comp=transpose(reshape(G_mat,settings.n_phi,settings.n_theta));
%         Gf_mat=Gf_col*reshape(transpose(f),(settings.n_theta)*settings.n_phi,1);
%     Gf_comp=transpose(reshape(Gf_mat,settings.n_phi,settings.n_theta));
%         Gb_mat=Gb_col*reshape(transpose(f),(settings.n_theta)*settings.n_phi,1);
%     Gb_comp=transpose(reshape(Gb_mat,settings.n_phi,settings.n_theta));

    %% Compensate for lost to keep integral equal 1
    rhs=rhs-settings.K_p*(Sf-1)*real(cf(:,1));
    rhs=rhs(2:(end-1),:);
    
    %% Output
    yp=reshape(transpose(rhs),(settings.n_theta-2)*settings.n_phi,1);
    
end
    