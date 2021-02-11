function Mlap=lap_mat(settings)

    n=settings.n;
    m=settings.m;
    Mtheta2 = kron((spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n))\...
        (spdiags((-n/2:n/2-1)'*1i,0, n, n)*...
        spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)*...
        spdiags((-n/2:n/2-1)'*1i,0, n, n)),speye(m));


        % This gives more accurate results in both time marching and
        % inversion.
        Mphi2 = kron(spdiags(0.5i*ones(n,1)*[-1,1], [-1 1], n, n)\...
            (spdiags(0.5i*ones(n,1)*[-1,1], [-1 1], n, n)\speye(n))...
            ,spdiags(-((-m/2:m/2-1).^2)',0, m, m));
        % This seems more natural, but would give spurious modes in
        % inversion and residue in time marching.
        % Mphi2 = kron((spdiags(.25*ones(n,1)*[-1,2,-1], [-2:2:2], n, n)\...
        %     speye(n)),spdiags(-((-m/2:m/2-1).^2)',0, m, m));

    Mlap=Mtheta2+Mphi2;
end