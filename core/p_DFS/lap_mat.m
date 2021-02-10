function Mlap=lap_mat(settings,inv)
    if nargin ~=2
        inv=0;
    end
    n=settings.n;
    m=settings.m;
    Mtheta2 = kron((spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n))\...
        (spdiags((-n/2:n/2-1)'*1i,0, n, n)*...
        spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)*...
        spdiags((-n/2:n/2-1)'*1i,0, n, n)),speye(m));


    if inv
        % This is better suited for direct inversion, as it eliminate spurious high modes
        Mphi2 = kron(spdiags(0.5i*ones(n,1)*[-1,1], [-1 1], n, n)\...
            (spdiags(0.5i*ones(n,1)*[-1,1], [-1 1], n, n)\speye(n))...
            ,spdiags(-((-m/2:m/2-1).^2)',0, m, m));
    else
        % This gives much better time stepping stability, but would give
        % spurious high modes if directly inverted
        Mphi2 = kron((spdiags(.25*ones(n,1)*[-1,2,-1], [-2:2:2], n, n)\...
            speye(n))...
            ,spdiags(-((-m/2:m/2-1).^2)',0, m, m));
    end

    Mlap=Mtheta2+Mphi2;
end