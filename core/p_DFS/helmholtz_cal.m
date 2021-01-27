function ucoeff = helmholtz_cal(Fcol, K2, helm)
% Forcing term:
F=transpose(reshape(Fcol,helm.n,helm.m));
int_const = helm.en*F(:,helm.k)/K2;

% Multiple rhs by sin(th)^2 and divide by K2:
F = helm.L2 * F / K2;

% Want to solve
%    X L^T + X DF^T = F
% subject to zero integral constraints, i.e.,  w^T X_0  = 0.

% Note that the matrix equation decouples because DF is diagonal.

% Solve decoupled matrix equation for X, one row at a time:
CFS = zeros(helm.m, helm.n);
L = (1/K2)*(helm.L1) + helm.L2;
scl = (helm.DF2ndiag)/K2;
for i = [floor(helm.n/2):-1:1 floor(helm.n/2)+2:helm.n]
    CFS(:,i) = (L + scl(i)*helm.Im) \ F(:,i);
end

% Now, do zeroth mode:
CFS(:, helm.k) = [L( 1:helm.floorm, :) ; helm.en ; L( helm.floorm+2:helm.m, :) ] ...
    \ [ F(1:helm.floorm, helm.k);int_const ; F(helm.floorm+2:helm.m, helm.k) ];

ucoeff=reshape(transpose(CFS),helm.n*helm.m,1);
end