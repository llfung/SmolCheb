function Madv=adv_gyro_mat(settings)
    
    n=settings.n;
    m=settings.m;
    Msinncosm = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m)); %e1
    Msinnsinm = kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n),spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m)); %e2
    Mcosn = kron(spdiags(.5*ones(n,1)*[1,1], [-1 1], n, n),speye(m)); %e3
    
%     invMsinn_sinm = kron(inv(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)),spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m)); 
    invMsinn_sinm = kron(inv(spdiags(ones(n,1)*[-1,1], [-1 1], n, n)),spdiags(ones(m,1)*[-1,1], [-1 1], m, m)); %for de1
%     invMsinn_cosm = kron(inv(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)),spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m)); 
    invMsinn_cosm = kron(inv(spdiags(1i*ones(n,1)*[-1,1], [-1 1], n, n)),spdiags(ones(m,1)*[1,1], [-1 1], m, m)); %for de2
    
    Mde1= -invMsinn_sinm*kron(speye(n),spdiags((-m/2:m/2-1)'*1i,0, m, m))...
        + kron(spdiags(.5*ones(n,1)*[1,1], [-1 1], n, n)*spdiags((-n/2:n/2-1)'*1i,0, n, n),...
        spdiags(.5*ones(m,1)*[1,1], [-1 1], m, m)); % de1
    Mde2= invMsinn_cosm*kron(speye(n),spdiags((-m/2:m/2-1)'*1i,0, m, m))...
        + kron(spdiags(.5*ones(n,1)*[1,1], [-1 1], n, n)*spdiags((-n/2:n/2-1)'*1i,0, n, n),...
        spdiags(.5i*ones(m,1)*[-1,1], [-1 1], m, m)); % de2
    Mde3= -kron(spdiags(.5i*ones(n,1)*[-1,1], [-1 1], n, n)*spdiags((-n/2:n/2-1)'*1i,0, n, n),speye(m)); % de3

    
    gyro1=-Mcosn*Msinncosm;
    gyro2=-Mcosn*Msinnsinm;
    gyro3=(speye(n*m)-Mcosn*Mcosn);
    
    Madv=(Mde1*gyro1+Mde2*gyro2+Mde3*gyro3);
    
end