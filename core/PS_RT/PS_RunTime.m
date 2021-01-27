classdef PS_RunTime
    %PS_RunTime RunTime PostProcessing Module
    %   Detailed explanation goes here
    
    properties
        varDir
        InvMeth
        saving_rate1
        saving_rate2
        settings
        
        dt
        ufull_save
        ucoeff_previous1
        ucoeff_previous2
        
        zero_row
        mats
        Linv
        g
        
        Transformed
    end
    
    methods
        function obj = PS_RunTime(variableDirection,InversionMethod,mats,settings,save_rate1,save_rate2)
            %InversionMethod Constructor
            %   Detailed explanation goes here
            obj.saving_rate1 = save_rate1;
            obj.dt=settings.dt;
            obj.settings=settings;
            obj.mats=mats;
            
            switch variableDirection
                case 'x'
                    obj.varDir=1;
                case 'z'
                    obj.varDir=2;
                otherwise
                    error('PS_RunTime: Variable Direction Incorrect');
            end
            
            switch InversionMethod
                case 'None'
                    obj.InvMeth=0;
                case 'inv'
                    obj.InvMeth=1;
                    obj.zero_row=zeros(1,obj.settings.N_mesh);
                    obj.Linv=NaN(obj.settings.n*obj.settings.m,...
                        obj.settings.n*obj.settings.m+1,obj.settings.N_mesh);
                    obj.g=NaN(...
                        obj.settings.n*obj.settings.m+1,obj.settings.N_mesh);
                    obj.Transformed.ex_g=NaN(1,obj.settings.N_mesh);
                    obj.Transformed.ez_g=NaN(1,obj.settings.N_mesh);
                    for j=1:obj.settings.N_mesh
                        Le=gather(mats.S_profile(j)*mats.Mvor+mats.Mgyro-mats.Mlap);
                        obj.Linv(:,:,j)=pinv([full(Le);full(gather(Mint))]);
                        obj.g(:,j)=obj.Linv(:,:,j)*[zeros(obj.settings.n*obj.settings.m);1/2/pi];
                        obj.Transformed.ex_g(j)=mats.Mint*(mats.Mp1*obj.g(:,j));
                        obj.Transformed.ez_g(j)=mats.Mint*(mats.Mp1*obj.g(:,j));
                    end
                case 'invGPU'
                    obj.InvMeth=2;
                    obj.zero_row=zeros(1,obj.settings.N_mesh,'gpuArrays');
                    obj.Linv=NaN(obj.settings.n*obj.settings.m,...
                        obj.settings.n*obj.settings.m+1,obj.settings.N_mesh,'gpuArray');
                    obj.g=NaN(...
                        obj.settings.n*obj.settings.m+1,obj.settings.N_mesh);
                    obj.Transformed.ex_g=NaN(1,obj.settings.N_mesh);
                    obj.Transformed.ez_g=NaN(1,obj.settings.N_mesh);
                    for j=1:obj.settings.N_mesh
                        Le=mats.S_profile(j)*mats.Mvor+mats.Mgyro-mats.Mlap;
                        obj.Linv(:,:,j)=pinv([full(Le);full(Mint)]);
                        obj.g(:,j)=obj.Linv(:,:,j)*[zeros(obj.settings.n*obj.settings.m);1/2/pi];
                        obj.Transformed.ex_g(j)=mats.Mint*(mats.Mp1*obj.g(:,j));
                        obj.Transformed.ez_g(j)=mats.Mint*(mats.Mp1*obj.g(:,j));
                    end
                case 'inv_w_fdt'
                    obj.InvMeth=3;
                    obj.zero_row=zeros(1,obj.settings.N_mesh);
                    obj.Linv=NaN(obj.settings.n*obj.settings.m,...
                        obj.settings.n*obj.settings.m+1,obj.settings.N_mesh);
                    obj.g=NaN(...
                        obj.settings.n*obj.settings.m+1,obj.settings.N_mesh);
                    obj.Transformed.ex_g=NaN(1,obj.settings.N_mesh);
                    obj.Transformed.ez_g=NaN(1,obj.settings.N_mesh);
                    for j=1:obj.settings.N_mesh
                        Le=gather(mats.S_profile(j)*mats.Mvor+mats.Mgyro-mats.Mlap);
                        obj.Linv(:,:,j)=pinv([full(Le);full(gather(Mint))]);
                        obj.g(:,j)=obj.Linv(:,:,j)*[zeros(obj.settings.n*obj.settings.m);1/2/pi];
                        obj.Transformed.ex_g(j)=mats.Mint*(mats.Mp1*obj.g(:,j));
                        obj.Transformed.ez_g(j)=mats.Mint*(mats.Mp1*obj.g(:,j));
                    end                    
                case 'invGPU_w_fdt'
                    obj.InvMeth=4;
                    obj.zero_row=zeros(1,obj.settings.N_mesh,'gpuArrays');
                    obj.Linv=NaN(obj.settings.n*obj.settings.m,...
                        obj.settings.n*obj.settings.m+1,obj.settings.N_mesh,'gpuArray');
                    obj.g=NaN(...
                        obj.settings.n*obj.settings.m+1,obj.settings.N_mesh);
                    obj.Transformed.ex_g=NaN(1,obj.settings.N_mesh);
                    obj.Transformed.ez_g=NaN(1,obj.settings.N_mesh);
                    for j=1:obj.settings.N_mesh
                        Le=mats.S_profile(j)*mats.Mvor+mats.Mgyro-mats.Mlap;
                        obj.Linv(:,:,j)=pinv([full(Le);full(Mint)]);
                        obj.g(:,j)=obj.Linv(:,:,j)*[zeros(obj.settings.n*obj.settings.m);1/2/pi];
                        obj.Transformed.ex_g(j)=mats.Mint*(mats.Mp1*obj.g(:,j));
                        obj.Transformed.ez_g(j)=mats.Mint*(mats.Mp1*obj.g(:,j));
                    end
                otherwise
                    error('PS_RunTime: Inversion Method incorrect');
            end
            if obj.InvMeth
                if obj.varDir==1
                    obj.Transformed.DDTxx=NaN(floor(obj.settings.nsteps/obj.saving_rate1),obj.settings.N_mesh);
                    obj.Transformed.DDTzx=NaN(floor(obj.settings.nsteps/obj.saving_rate1),obj.settings.N_mesh);
                else
                    obj.Transformed.DDTxz=NaN(floor(obj.settings.nsteps/obj.saving_rate1),obj.settings.N_mesh);
                    obj.Transformed.DDTzz=NaN(floor(obj.settings.nsteps/obj.saving_rate1),obj.settings.N_mesh);
                end
                obj.Transformed.Dxx=NaN(floor(obj.settings.nsteps/obj.saving_rate1),obj.settings.N_mesh);
                obj.Transformed.Dxz=NaN(floor(obj.settings.nsteps/obj.saving_rate1),obj.settings.N_mesh);
                obj.Transformed.Dzx=NaN(floor(obj.settings.nsteps/obj.saving_rate1),obj.settings.N_mesh);
                obj.Transformed.Dzz=NaN(floor(obj.settings.nsteps/obj.saving_rate1),obj.settings.N_mesh);
                obj.Transformed.Vix=NaN(floor(obj.settings.nsteps/obj.saving_rate1),obj.settings.N_mesh);
                obj.Transformed.Viz=NaN(floor(obj.settings.nsteps/obj.saving_rate1),obj.settings.N_mesh);
                obj.Transformed.VDTx=NaN(floor(obj.settings.nsteps/obj.saving_rate1),obj.settings.N_mesh);
                obj.Transformed.VDTz=NaN(floor(obj.settings.nsteps/obj.saving_rate1),obj.settings.N_mesh);
            end
            if obj.InvMeth>2
                obj.Transformed.Vux=NaN(floor((obj.settings.nsteps-2)/obj.saving_rate1),obj.settings.N_mesh);
                obj.Transformed.Vuz=NaN(floor((obj.settings.nsteps-2)/obj.saving_rate1),obj.settings.N_mesh);
                obj.ucoeff_previous1=NaN(obj.settings.n*obj.settings.m,obj.settings.N_mesh,3);
            end
            if nargin>5
                obj.saving_rate2 = save_rate2;
            else
                obj.saving_rate2 = Inf;
            end
            if ~isinf(obj.saving_rate2)
                obj.ucoeff_previous2=NaN(obj.settings.n*obj.settings.m,obj.settings.N_mesh,3);
            end
        end
        
        function RunTimeCall(obj,ucoeff,Mint,i)
            %RunTimeCall To be called at every time step
            %   Detailed explanation goes here
            %% Saving rate 1 - Transforming
            if obj.InvMeth
            if ( mod(i, obj.saving_rate1) == 0 )
                f=ucoeff/real(Mint*ucoeff*2*pi);
                if obj.varDir==1
                    [Dxx,Dxz,Dzx,Dzz,Vix,Viz,VDTx,VDTz,DDTxx,DDTzx]=...
                        Linv_f_x(f,obj.Linv,obj.mats.Rdx,obj.mats.Rd2x,obj.mats.Mp1,obj.mats.Mp3,obj.settings,obj.zero_row);
                    obj.Transformed.DDTxx(i/obj.saving_rate1,:)=DDTxx;
                    obj.Transformed.DDTzx(i/obj.saving_rate1,:)=DDTzx;
                else
                    [Dxx,Dxz,Dzx,Dzz,Vix,Viz,VDTx,VDTz,DDTxz,DDTzz]=...
                        Linv_f_z(f,obj.Linv,obj.mats.Rdz,obj.mats.Rd2z,obj.mats.Mp1,obj.mats.Mp3,obj.settings,obj.zero_row);
                    obj.Transformed.DDTxz(i/obj.saving_rate1,:)=DDTxz;
                    obj.Transformed.DDTzz(i/obj.saving_rate1,:)=DDTzz;
                end
                obj.Transformed.Dxx(i/obj.saving_rate1,:)  =Dxx;
                obj.Transformed.Dxz(i/obj.saving_rate1,:)  =Dxz;
                obj.Transformed.Dzx(i/obj.saving_rate1,:)  =Dzx;
                obj.Transformed.Dzz(i/obj.saving_rate1,:)  =Dzz;
                obj.Transformed.Vix(i/obj.saving_rate1,:)  =Vix;
                obj.Transformed.Viz(i/obj.saving_rate1,:)  =Viz;
                obj.Transformed.VDTx(i/obj.saving_rate1,:) =VDTx;
                obj.Transformed.VDTz(i/obj.saving_rate1,:) =VDTz;

            end
            if obj.InvMeth>2
                if ( mod(i, obj.saving_rate1) == 2 )&& i~=2
                    fdt=((-ucoeff./(real(Mint*ucoeff*2*pi))...
                        + obj.ucoeff_previous1(:,:,1)./(real(Mint*obj.ucoeff_previous1(:,:,1)*2*pi)))/12 ...
                        +(obj.ucoeff_previous1(:,:,3)./(real(Mint*obj.ucoeff_previous1(:,:,3)*2*pi))...
                        - obj.ucoeff_previous1(:,:,2)./(real(Mint*obj.ucoeff_previous1(:,:,2)*2*pi)))*(2/3))/obj.dt;
                    
                    [Vux,Vuz]=Linv_fdt(fdt,obj.Linv,obj.mat.Mp1,obj.mat.Mp3,obj.settings,obj.zero_row);
                    obj.Transformed.Vux(i/obj.saving_rate1,:) =Vux;
                    obj.Transformed.Vuz(i/obj.saving_rate1,:) =Vuz;
                end
                if ( mod(i, obj.saving_rate1) == 1 )&& i~=1
                    obj.ucoeff_previous1(:,:,3)=ucoeff;
                end
                if ( mod(i, obj.saving_rate1) == obj.saving_rate1-1 )
                    obj.ucoeff_previous1(:,:,2)=ucoeff;
                end
                if ( mod(i, obj.saving_rate1) == obj.saving_rate1-2 )
                    obj.ucoeff_previous1(:,:,1)=ucoeff;
                end
            end
            end
            
            %% Saving rate 2 - Saving into mat
            if ( mod(i, obj.saving_rate2) == 0 )
                obj.ufull_save=ucoeff;
                t=i*obj.dt;
            end
            if ( mod(i, obj.saving_rate2) == 2 )&& i~=2
                fdt_full_save=((-ucoeff./(real(Mint*ucoeff*2*pi))...
                    + obj.ucoeff_previous2(:,:,1)./(real(Mint*obj.ucoeff_previous2(:,:,1)*2*pi)))/12 ...
                    +(obj.ucoeff_previous2(:,:,3)./(real(Mint*obj.ucoeff_previous2(:,:,3)*2*pi))...
                    -obj.ucoeff_previous2(:,:,2)./(real(Mint*obj.ucoeff_previous2(:,:,2)*2*pi)))*(2/3))/obj.dt;
                udt_full_save=((-ucoeff...
                    + obj.ucoeff_previous2(:,:,1))/12 ...
                    +(obj.ucoeff_previous2(:,:,3)...
                    -obj.ucoeff_previous2(:,:,2))*(2/3))/obj.dt;
                ufull_save=obj.ufull_save; %#ok<PROPLC>
                save(['t' num2str(t) '.mat'],'t','ufull_save','fdt_full_save','udt_full_save');
            end
            if ( mod(i, obj.saving_rate2) == 1 )&& i~=1
                obj.ucoeff_previous2(:,:,3)=ucoeff;
            end
            if ( mod(i, obj.saving_rate2) == obj.saving_rate2-1 )
                obj.ucoeff_previous2(:,:,2)=ucoeff;
            end
            if ( mod(i, obj.saving_rate2) == obj.saving_rate2-2 )
                obj.ucoeff_previous2(:,:,1)=ucoeff;
            end
        end
    end
end

