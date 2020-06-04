function yp=full_phy_rep_y_op(~,y,settings)

    %% Initialization
    zi=sqrt(-1);
    
    %% Phi space FFT calculation
    f_wobc=transpose(reshape(y,settings.n_phi,settings.N_mesh*(settings.n_theta-2)));
    n_theta_wobc=(settings.n_theta-2);
    %% 2nd order BC implementation
    for i=1:settings.N_mesh
    f_theta_0=ones(1,settings.n_phi)*mean(18*f_wobc((i-1)*n_theta_wobc+1,:)-9*f_wobc((i-1)*n_theta_wobc+2,:)+2*f_wobc((i-1)*n_theta_wobc+3,:))/11;
    f_theta_n=ones(1,settings.n_phi)*mean(18*f_wobc(    i*n_theta_wobc-1,:)-9*f_wobc(    i*n_theta_wobc-2,:)+2*f_wobc(    i*n_theta_wobc-3,:))/11;
    f=[f_theta_0;f_wobc((i-1)*n_theta_wobc+1:i*n_theta_wobc,:);f_theta_n];
    
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

    MM=transpose(spdiags([sin(settings.theta)/(12*settings.dtheta),...
        -2*sin(settings.theta)/(3*settings.dtheta),...
        -2*cos(settings.theta),...
        2*sin(settings.theta)/(3*settings.dtheta),...
        -sin(settings.theta)/(12*settings.dtheta),...
        ],...
        -2:2,settings.n_theta,settings.n_theta));
    
    MM=[zeros(1,settings.n_theta);...
        MM(2:(settings.n_theta-1),:);...
        zeros(1,settings.n_theta)];
    gg=settings.beta*MM*f;

    cot_term=[0;cot(settings.theta(2:(settings.n_theta-1)));0];

    omg=settings.S/2*(cot_term.*df.*(settings.omega_e1*cos(settings.phi)+...
        settings.omega_e3*sin(settings.phi))-settings.omega_e2*df+...
        transpose(spdiags(-ones(settings.n_theta,1)*...
        [1/12,-2/3,0,2/3,-1/12],...
        -2:2,settings.n_theta,settings.n_theta))*f.*...
        (settings.omega_e1*sin(settings.phi)...
        -settings.omega_e3*cos(settings.phi))/(settings.dtheta));

    rx=gg+omg;
    
    pre_ll=transpose(...
        spdiags(-cot(settings.theta)*[1/12,-2/3,0,2/3,-1/12]/settings.dtheta...
        +ones(settings.n_theta,1)*[-1/12,4/3,-5/2,4/3,-1/12]*settings.dtheta^(-2),...
        -2:2,settings.n_theta,settings.n_theta));
    pre_ll=[zeros(1,settings.n_theta);...
        pre_ll(2:(settings.n_theta-1),:);...
        zeros(1,settings.n_theta)];
    ll=pre_ll*f...
        +([0;sin(settings.theta(2:(settings.n_theta-1))).^(-2);0]).*d2f;

    rhs=ll-rx;
    
    %% Compensate for lost to keep integral equal 1
    rhs=rhs-settings.K_p*(Sf-0.5)*real(cf(:,1));
    rhs_wobc=rhs(2:(end-1),:);
    
    %% Output
    yp((i-1)*n_theta_wobc*settings.n_phi+1:i*n_theta_wobc*settings.n_phi,1)=reshape(transpose(rhs_wobc),(settings.n_theta-2)*settings.n_phi,1);
    end
end
    