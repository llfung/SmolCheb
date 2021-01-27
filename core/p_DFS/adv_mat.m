function Madv=adv_mat(settings)
persistent n m Mvor1 Mvor2 Mvor3 Mgyro

if isempty(n)
    n=settings.n;
    m=settings.m;
    [Mvor1,Mvor2,Mvor3,Mgyro]=gen_mat(n,m);
elseif ~(settings.n==n && settings.m==m)
    n=settings.n;
    m=settings.m;
    [Mvor1,Mvor2,Mvor3,Mgyro]=gen_mat(n,m);
end
    
    Madv=settings.S*(settings.omg1*Mvor1+settings.omg2*Mvor2+settings.omg3*Mvor3)...
        +settings.beta*Mgyro;  
end
function [Mvor1,Mvor2,Mvor3,Mgyro]=gen_mat(n,m)

    M_n_cotn = (spdiags(ones(n,1)*[-1,1], [-1 1], n, n))\spdiags(.5*ones(n,1)*[1,1], [-1 1], n, n);
    
    Mvor1 = -kron(spdiags((-n/2:n/2-1)'*1i,0, n, n),spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m))...
        -kron(M_n_cotn,spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m)*spdiags((-m/2:m/2-1)'*1i,0, m, m));
    Mvor2 = kron(spdiags((-n/2:n/2-1)'*1i,0, n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m))...
        -kron(M_n_cotn,spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m)*spdiags((-m/2:m/2-1)'*1i,0, m, m));
    Mvor3 = -kron(spdiags((-n/2:n/2-1)'*1i,0, n, n),speye(m));
    
    Mgyro=kron(-spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)*spdiags((-n/2:n/2-1)'*1i,0, n, n)...
        -2*spdiags(.5*ones(n,1)*[1,1], [-1 1], n, n),speye(m));
end