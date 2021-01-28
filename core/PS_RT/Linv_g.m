function [g,Linv]=Linv_g(S_profile,Mvor,Mgyro,Mlap,Mint)
    N_mesh=numel(S_profile);
    nm=size(Mvor,2);
    g=NaN(nm,N_mesh);
    Linv=NaN(nm,nm+1,N_mesh);
    for j=1:N_mesh
        Le=gather(S_profile(j)*Mvor+Mgyro-Mlap);
        Linv(:,:,j)=pinv([full(Le);full(gather(Mint))]);
        g(:,j)=Linv(:,:,j)*[zeros(nm,1);1/2/pi];
    end
end