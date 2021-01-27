function [Dxx,Dxz,Dzx,Dzz,Vix,Viz,VDTx,VDTz,DDTxz,DDTzz]=Linv_f_z(f,Linv,Rd,Rd2,Mp1,Mp3,settings,zero_row)
        d2f=f*Rd2;
        df=f*Rd;
        ex_avg=real(Mint*Mp1*f*(2*pi));
        ez_avg=real(Mint*Mp3*f*(2*pi));
        
        bx_RHS=Mp1*f-ex_avg.*f;
        bz_RHS=Mp3*f-ez_avg.*f;
%        inhomo_RHS=Mp1*(f*Rd)-(ex_avg*Rd).*f;
        inhomo_RHS=Mp3*(f*Rd)-(ez_avg*Rd).*f;
        
        temp=sum(bsxfun(@times,Linv,reshape([bx_RHS;zeros(1,N_mesh,'gpuArray')],1,n*m+1,N_mesh)),2);
        bx=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        temp=sum(bsxfun(@times,Linv,reshape([bz_RHS;zeros(1,N_mesh,'gpuArray')],1,n*m+1,N_mesh)),2);
        bz=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        temp=sum(bsxfun(@times,Linv,reshape([df;zeros(1,N_mesh,'gpuArray')],1,n*m+1,N_mesh)),2);
        b_DT=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        temp=sum(bsxfun(@times,Linv,reshape([inhomo_RHS;zeros(1,N_mesh,'gpuArray')],1,n*m+1,N_mesh)),2);
        f_inhomo=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        temp=sum(bsxfun(@times,Linv,reshape([d2f;zeros(1,N_mesh,'gpuArray')],1,n*m+1,N_mesh)),2);
        f_a=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        
        Dxx=Mint*(Mp1*reshape(bx,n*m,N_mesh))*(2*pi);
        Dxz=Mint*(Mp1*reshape(bz,n*m,N_mesh))*(2*pi);
        Dzx=Mint*(Mp3*reshape(bx,n*m,N_mesh))*(2*pi);
        Dzz=Mint*(Mp3*reshape(bz,n*m,N_mesh))*(2*pi);
        Vix=Mint*(Mp1*reshape(f_inhomo,n*m,N_mesh))*(2*pi);
        Viz=Mint*(Mp3*reshape(f_inhomo,n*m,N_mesh))*(2*pi);

         VDTx=Mint*(Mp1*reshape(f_a,n*m,N_mesh))*(2*pi);
         VDTz=Mint*(Mp3*reshape(f_a,n*m,N_mesh))*(2*pi);
        DDTxz=Mint*(Mp1*reshape(b_DT,n*m,N_mesh))*(2*pi);
        DDTzz=Mint*(Mp3*reshape(b_DT,n*m,N_mesh))*(2*pi);
end