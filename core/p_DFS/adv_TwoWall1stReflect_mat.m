function M=adv_TwoWall1stReflect_mat(settings)
% Numerical issue with this!!! singularity near theta=0/pi
% Wall is at z= +-H

    n=settings.n;
    m=settings.m;
    H=settings.H; % Channel Half-Width
    h=settings.h; % Distance from centre (<H)
    B=settings.B;
    A1=settings.A1;
    A2=settings.A2;
    A3=settings.A3;

    Mcosm  = spdiags(.5 *ones(m,1)*[ 1,1], [-1 1], m, m); %cos phi
    Msinm  = spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m); %sin phi
    Mcosn  = spdiags(.5 *ones(n,1)*[ 1,1], [-1 1], n, n); %cos theta
    Msinn  = spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n); %sin theta
    
%     Mcos2n = spdiags(.5 *ones(n,1)*[ 1,1], [-2 2], n, n); %cos 2theta
    Mcos2n = 2*Mcosn*Mcosn-speye(n);

    Mcos4n = Mcosn*Mcosn*Mcosn*Mcosn-6*Mcosn*Mcosn*Msinn*Msinn+Msinn*Msinn*Msinn*Msinn;
    Mcos6n = Mcosn*Mcosn*Mcosn*Mcosn*Mcosn*Mcosn-15*Mcosn*Mcosn*Mcosn*Mcosn*Msinn*Msinn...
        +15*Mcosn*Mcosn*Msinn*Msinn*Msinn*Msinn-Msinn*Msinn*Msinn*Msinn*Msinn*Msinn;

    Prefac = 3*settings.AR^3*H*(3*h^2+H^2)/8192/(h^2-H^2)^3/pi;
    
    M_ = kron(-16*Mcosn*Msinn*(24*A1 - 16*A2 - 8*A3 + 27*A1*B + 244*A2*B + 17*A3*B ...
        + 8*(A3*(5 - 11*B) - A2*(20 + B) + 3*A1*(5 + 4*B))*Mcos2n ...
        + 7*(3*A1 - 4*A2 + A3)*B*Mcos4n),Mcosm) ...
        - kron(...
        128*Msinn\Mcosn*(A3*(-1 + 2*B) ... 
        + A2*(-1 + 4*B) + (A2 - A3)*(-1 + 2*B)*Mcos2n),Msinm*spdiags((-m/2:m/2-1)'*1i,0, m, m))...
        - kron(...
         4*(24*A1 + 32*A2 + 8*A3 + 18*A1*B - 18*A3*B + (3*A1 - 196*A2 + A3)*B*Mcos2n ...
        -2*(12*A1 - 16*A2 + 4*A3 + 9*A1*B - 9*A3*B)*Mcos4n ...
        +(-3*A1*B+4*A2*B-A3*B)*Mcos6n)...
        *spdiags((-n/2:n/2-1)'*1i,0, n, n),Mcosm);
    M = Prefac*(M_);
end