function Madv=adv_strain_mat(settings)
    
    n=settings.n;
    m=settings.m;
    M_n_cotn = (spdiags(1i*ones(n,1)*[-1,1], [-1 1], n, n))\spdiags(ones(n,1)*[1,1], [-1 1], n, n);
    diagEterm = settings.e11+settings.e22-settings.e33*2;
    Mcosm  = spdiags(.5 *ones(m,1)*[ 1,1], [-1 1], m, m); %cos phi
    Msinm  = spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m); %sin phi
%     Mcosn  = spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n); %cos theta
%     Msinn  = spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n); %sin theta
    Mcos2m = spdiags(.5 *ones(m,1)*[ 1,1], [-2 2], m, m); %cos 2phi
    Msin2m = spdiags(.5i*ones(m,1)*[-1,1], [-2 2], m, m); %sin 2phi
    Mcos2n = spdiags(.5 *ones(n,1)*[ 1,1], [-2 2], n, n); %cos 2theta
    Msin2n = spdiags(.5i*ones(n,1)*[-1,1], [-2 2], n, n); %sin 2theta
    Msinsqn = spdiags(.5*ones(n,1)*[-1/2,1,-1/2], [-2 0 2], n, n); %sin 2theta 
    
    Madv = (diagEterm*(speye(m*n)+3*kron(Mcos2n,speye(m)))...
        -12*kron(Msin2n,settings.e13*Mcosm+settings.e23*Msinm)...
        -6*kron(Msinsqn,(settings.e11-settings.e22)*Mcos2m+2*settings.e12*Msin2m))/4;
    Madv = Madv+(...
        kron(speye(n),settings.e12*Mcos2m+(settings.e22-settings.e11)/2*Msin2m)...
        +kron(M_n_cotn,settings.e23*Mcosm-settings.e13*Msinm)...
        )*kron(speye(n),spdiags((-m/2:m/2-1)'*1i,0, m, m));
    Madv = Madv+(...
        kron(Mcos2n,settings.e13*Mcosm+settings.e23*Msinm)...
        +kron(Msin2n/4,diagEterm*speye(m)+(settings.e11-settings.e22)*Mcos2m+2*settings.e12*Msin2m)...
        )*kron(spdiags((-n/2:n/2-1)'*1i,0, n, n),speye(m));

end