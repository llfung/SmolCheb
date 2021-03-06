function [ex_avg,ez_avg,Dxx,Dxz,Dzx,Dzz,Vix,Viz,VDTx,VDTz,DDTx,DDTz]=Linv_f(dir,f,Linv,Msin,Rd,Rd2,Mp1,Mp3,settings,zero_row,k)
        N_mesh=settings.N_mesh;
        n=settings.n;
        m=settings.m;
        Mint=settings.Mint;
        
        d2f=f*Rd2;
        df=f*Rd;
        ex_avg=real(Mint*Mp1*f*(2*pi));
        ez_avg=real(Mint*Mp3*f*(2*pi));
        
        bx_RHS=Mp1*f-ex_avg.*f;
        bz_RHS=Mp3*f-ez_avg.*f;
        switch dir
        case 'x'
          inhomo_RHS=Mp1*(f*Rd)-(ex_avg*Rd).*f;
        case 'z'
          inhomo_RHS=Mp3*(f*Rd)-(ez_avg*Rd).*f;
        otherwise
          error('Linv_f: unknown direction');
        end
        
        bx_RHS=Msin*bx_RHS;bx_RHS(k,:)=zero_row;
        temp=sum(bsxfun(@times,Linv,reshape(bx_RHS,1,n*m,N_mesh)),2);
        bx=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        
        bz_RHS=Msin*bz_RHS;bz_RHS(k,:)=zero_row;
        temp=sum(bsxfun(@times,Linv,reshape(bz_RHS,1,n*m,N_mesh)),2);
        bz=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        
        df=Msin*df;df(k,:)=zero_row;
        temp=sum(bsxfun(@times,Linv,reshape(df,1,n*m,N_mesh)),2);
        b_DT=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        
        inhomo_RHS=Msin*inhomo_RHS;inhomo_RHS(k,:)=zero_row;
        temp=sum(bsxfun(@times,Linv,reshape(inhomo_RHS,1,n*m,N_mesh)),2);
        f_inhomo=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        
        d2f=Msin*d2f;d2f(k,:)=zero_row;
        temp=sum(bsxfun(@times,Linv,reshape(d2f,1,n*m,N_mesh)),2);
        f_a=reshape(temp(1:n*m,1,:),n*m,N_mesh);
        
        Dxx=real(Mint*(Mp1*reshape(bx,n*m,N_mesh))*(2*pi));
        Dxz=real(Mint*(Mp1*reshape(bz,n*m,N_mesh))*(2*pi));
        Dzx=real(Mint*(Mp3*reshape(bx,n*m,N_mesh))*(2*pi));
        Dzz=real(Mint*(Mp3*reshape(bz,n*m,N_mesh))*(2*pi));
        Vix=real(Mint*(Mp1*reshape(f_inhomo,n*m,N_mesh))*(2*pi));
        Viz=real(Mint*(Mp3*reshape(f_inhomo,n*m,N_mesh))*(2*pi));

         VDTx=real(Mint*(Mp1*reshape(f_a,n*m,N_mesh))*(2*pi));
         VDTz=real(Mint*(Mp3*reshape(f_a,n*m,N_mesh))*(2*pi));
        DDTx=real(Mint*(Mp1*reshape(b_DT,n*m,N_mesh))*(2*pi));
        DDTz=real(Mint*(Mp3*reshape(b_DT,n*m,N_mesh))*(2*pi));
end