function [g,Linv]=Linv_g(S_profile,Mvor,Mgyro,Mlap,Mint,Msin,n,m)
    N_mesh=numel(S_profile);
    nm=n*m;
    g=NaN(nm,N_mesh);
    Linv=NaN(nm,nm,N_mesh);
    for j=1:N_mesh
        Le=full(Msin*gather(S_profile(j)*Mvor+Mgyro-Mlap));
        Le(n*m/2+m/2+1,:)=Mint;
        
        Linv(:,:,j)=inv(Le);
        
        g(:,j)=Linv(:,:,j)*[zeros(n*m/2+m/2,1);1/2/pi;zeros(n*m/2-m/2-1,1)];
    end
end