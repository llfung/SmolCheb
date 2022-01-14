function Madv=adv_inertial_VminVmax_mat(settings)
    
    n=settings.n;
    m=settings.m;

    Mcosn  = spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n); %cos theta
    Msinn  = spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n); %sin theta
    
    Madv = kron(speye(n)-3*Mcosn*Mcosn-Mcosn*Msinn*spdiags((-n/2:n/2-1)'*1i,0, n, n),speye(m));

end