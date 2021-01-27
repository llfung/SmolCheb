function Madv=adv_vor_mat(settings)
% persistent n m Msinncosm Msinnsinm Mcosn Mde1 Mde2 Mde3
% 
% if isempty(n) || ~(settings.n==n && settings.m==m)
    
    n=settings.n;
    m=settings.m;
    M_n_cotn = (spdiags(1i*ones(n,1)*[-1,1], [-1 1], n, n))\spdiags(ones(n,1)*[1,1], [-1 1], n, n);
    
    Mvor1 = -kron(spdiags((-n/2:n/2-1)'*1i,0, n, n),spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m))...
        -kron(M_n_cotn,spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m)*spdiags((-m/2:m/2-1)'*1i,0, m, m));
    Mvor2 = kron(spdiags((-n/2:n/2-1)'*1i,0, n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m))...
        -kron(M_n_cotn,spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m)*spdiags((-m/2:m/2-1)'*1i,0, m, m));
    Mvor3 = kron(speye(n),spdiags((-m/2:m/2-1)'*1i,0, m, m));
% end

    Madv=(settings.omg1*Mvor1+settings.omg2*Mvor2+settings.omg3*Mvor3);
end