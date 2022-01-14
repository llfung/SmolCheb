function Madv=adv_inertial_VminVc_mat(settings)
    
    n=settings.n;
    m=settings.m;

    Mcosn  = spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n); %cos theta
    Msinn  = spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n); %sin theta
    
%     Mcos2n = spdiags(.5 *ones(n,1)*[ 1,1], [-2 2], n, n); %cos 2theta
    Mcos2n = 2*Mcosn*Mcosn-speye(n);

    Madv = kron(-Mcos2n-Mcosn*Msinn*spdiags((-n/2:n/2-1)'*1i,0, n, n),speye(m));

end