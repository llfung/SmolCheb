function Madv=adv_gyro_mat(settings)

    n=settings.n;
    m=settings.m;

    Madv = kron(-spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)*spdiags((-n/2:n/2-1)'*1i,0, n, n)...
        -2*spdiags(.5*ones(n,1)*[1,1], [-1 1], n, n),speye(m));
    
end