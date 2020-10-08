function out=eavg(f,e_all_field,integrad)
%     global theta phi
    R1=f.*e_all_field(:,:,1);
    R2=f.*e_all_field(:,:,2);
    R3=f.*e_all_field(:,:,3);

    out.e1=integrad*mean(R1,2);
    out.e2=integrad*mean(R2,2);
    out.e3=integrad*mean(R3,2);
    
end
    