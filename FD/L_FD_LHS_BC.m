function LHS=L_FD_LHS_BC(LHS,settings)
% Applying Boundary Condition for LHS Matrix for Operator (-L) for solving f(e)
%   By assuming df\dtheta=0 at theta=0,pi, in 6th order,
%   using average over phi 

    BC1=kron(sparse([-49/20 zeros(1,settings.n_theta-1);...
        zeros(settings.n_theta-2,settings.n_theta);...
        zeros(1,settings.n_theta-1) 49/20]),speye(settings.n_phi))/settings.dtheta;
    BC2=kron(sparse([0 6 -15/2 20/3 -15/4 6/5 -1/6 zeros(1,settings.n_theta-7);...
        zeros(settings.n_theta-2,settings.n_theta);...
        zeros(1,settings.n_theta-7) 1/6 -6/5 15/4 -20/3 15/2 -6 0]),sparse(ones(settings.n_phi,settings.n_phi)))/settings.dtheta/settings.n_phi;

    LHS(1:settings.n_phi,:)=0;
    LHS(settings.n_phi*(settings.n_theta-1)+1:settings.n_phi*(settings.n_theta),:)=0;
    
    LHS=LHS+BC1+BC2;
end