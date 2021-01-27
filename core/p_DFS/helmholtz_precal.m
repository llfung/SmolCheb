function LHSinv = helmholtz_precal( K2, helm)
% Want to solve
%    X L^T + X DF^T = F
% subject to zero integral constraints, i.e.,  w^T X_0  = 0.

% Note that the matrix equation decouples because DF is diagonal.

% Solve decoupled matrix equation for X, one row at a time:
LHSinv = sparse(helm.m*helm.n, helm.m*helm.n);
L = (1/K2)*(helm.L1) + helm.L2;
scl = helm.DF2ndiag/K2;
for i = [floor(helm.n/2):-1:1 floor(helm.n/2)+2:helm.n]
%     LHSinv((i-1)*helm.m+1:i*helm.m,(i-1)*helm.m+1:i*helm.m) = (1/K2)*(full(L + scl(i)*helm.Im))\helm.L2;
    LHSinv((i-1)*helm.m+1:i*helm.m,(i-1)*helm.m+1:i*helm.m) = inv(full(L + scl(i)*helm.Im));
end

% Now, do zeroth mode:
% int_mat=speye(helm.m);
% int_mat(helm.floorm+1,:)=(1/K2)*helm.enG;
% LHSinv((helm.k-1)*helm.m+1:helm.k*helm.m,(helm.k-1)*helm.m+1:helm.k*helm.m)  = (1/K2)*(full((1/K2)*(helm.L1_zeroth) + helm.L2_zeroth))\helm.L2G*int_mat;

LHSinv((helm.k-1)*helm.m+1:helm.k*helm.m,(helm.k-1)*helm.m+1:helm.k*helm.m)  = inv(full((1/K2)*(helm.L1_zeroth) + helm.L2_zeroth));

end