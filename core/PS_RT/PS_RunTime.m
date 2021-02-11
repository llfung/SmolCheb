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
        Msin
        g
        
        Transformed
    end
    
    methods
        function obj = PS_RunTime(variableDirection,InversionMethod,mats,settings,save_rate1,save_rate2)
            %InversionMethod Constructor
            %   Detailed explanation goes here
            narginchk(5,6);
            
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
                case 'invGPU'
                    obj.InvMeth=2;
                    obj.zero_row=zeros(1,obj.settings.N_mesh,'gpuArray');
                case 'inv_w_fdt'
                    obj.InvMeth=3;
                    obj.zero_row=zeros(1,obj.settings.N_mesh);             
                case 'invGPU_w_fdt'
                    obj.InvMeth=4;
                    obj.zero_row=zeros(1,obj.settings.N_mesh,'gpuArray');
                otherwise
                    error('PS_RunTime: Inversion Method incorrect');
            end

            if obj.InvMeth
                Msin=kron(spdiags(0.5i*ones(settings.n,1)*[-1,1], [-1 1], settings.n, settings.n),speye(settings.m));
                [g,Linv]=Linv_g(mats.S_profile,mats.Mvor,mats.Mgyro,mats.Mlap,settings.Mint,Msin,settings.n,settings.m);
                obj.g=g;
                if mod(obj.InvMeth,2)
                    obj.Linv=Linv;
                    obj.Msin=Msin;
                else
                    obj.Linv=gpuArray(Linv);
                    obj.Msin=gpuArray(Msin);
                end
                if obj.varDir==1
                    [obj.Transformed.ex_g,obj.Transformed.ez_g,...
                        obj.Transformed.Dxx_g,obj.Transformed.Dxz_g,obj.Transformed.Dzx_g,obj.Transformed.Dzz_g,...
                        obj.Transformed.Vix_g,obj.Transformed.Viz_g,obj.Transformed.VDTx_g,obj.Transformed.VDTz_g,...
                        obj.Transformed.DDTxx_g,obj.Transformed.DDTzx_g]=...
                        Linv_f('x',g,obj.Linv,obj.Msin,mats.Rdx,mats.Rd2x,mats.Mp1,mats.Mp3,settings,zeros(1,settings.N_mesh),settings.n*settings.m/2+settings.m/2+1);
                else
                    [obj.Transformed.ex_g,obj.Transformed.ez_g,...
                        obj.Transformed.Dxx_g,obj.Transformed.Dxz_g,obj.Transformed.Dzx_g,obj.Transformed.Dzz_g,...
                        obj.Transformed.Vix_g,obj.Transformed.Viz_g,obj.Transformed.VDTx_g,obj.Transformed.VDTz_g,...
                        obj.Transformed.DDTxz_g,obj.Transformed.DDTzz_g]=...
                        Linv_f('z',g,obj.Linv,obj.Msin,mats.Rdz,mats.Rd2z,mats.Mp1,mats.Mp3,settings,zeros(1,settings.N_mesh),settings.n*settings.m/2+settings.m/2+1);
                end
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
        
        function obj=RunTimeCall(obj,ucoeff,i)
            %RunTimeCall To be called at every time step
            %   Detailed explanation goes here
            %% Saving rate 1 - Transforming
            if obj.InvMeth
            if ( mod(i, obj.saving_rate1) == 0 )
                if mod(obj.InvMeth,2)
                    f=gather(ucoeff./real(obj.settings.Mint*ucoeff*2*pi));
                else
                    f=ucoeff./real(obj.settings.Mint*ucoeff*2*pi);
                end
                if obj.varDir==1
                    [ex,ez,Dxx,Dxz,Dzx,Dzz,Vix,Viz,VDTx,VDTz,DDTxx,DDTzx]=...
                        Linv_f('x',f,obj.Linv,obj.Msin,obj.mats.Rdx,obj.mats.Rd2x,obj.mats.Mp1,obj.mats.Mp3,obj.settings,obj.zero_row,obj.settings.n*obj.settings.m/2+obj.settings.m/2+1);
                    obj.Transformed.DDTxx(i/obj.saving_rate1,:)=gather(DDTxx);
                    obj.Transformed.DDTzx(i/obj.saving_rate1,:)=gather(DDTzx);
                else
                    [ex,ez,Dxx,Dxz,Dzx,Dzz,Vix,Viz,VDTx,VDTz,DDTxz,DDTzz]=...
                        Linv_f('z',f,obj.Linv,obj.Msin,obj.mats.Rdz,obj.mats.Rd2z,obj.mats.Mp1,obj.mats.Mp3,obj.settings,obj.zero_row,obj.settings.n*obj.settings.m/2+obj.settings.m/2+1);
                    obj.Transformed.DDTxz(i/obj.saving_rate1,:)=gather(DDTxz);
                    obj.Transformed.DDTzz(i/obj.saving_rate1,:)=gather(DDTzz);
                end
                obj.Transformed.Dxx(i/obj.saving_rate1,:)  =gather(Dxx);
                obj.Transformed.Dxz(i/obj.saving_rate1,:)  =gather(Dxz);
                obj.Transformed.Dzx(i/obj.saving_rate1,:)  =gather(Dzx);
                obj.Transformed.Dzz(i/obj.saving_rate1,:)  =gather(Dzz);
                obj.Transformed.Vix(i/obj.saving_rate1,:)  =gather(Vix);
                obj.Transformed.Viz(i/obj.saving_rate1,:)  =gather(Viz);
                obj.Transformed.VDTx(i/obj.saving_rate1,:) =gather(VDTx);
                obj.Transformed.VDTz(i/obj.saving_rate1,:) =gather(VDTz);
                
                obj.Transformed.ex(i/obj.saving_rate1,:) =gather(ex);
                obj.Transformed.ez(i/obj.saving_rate1,:) =gather(ez);
            end
            if obj.InvMeth>2
                if ( mod(i, obj.saving_rate1) == 2 )&& i~=2
                    fdt=((-ucoeff./(real(obj.settings.Mint*ucoeff*2*pi))...
                        + obj.ucoeff_previous1(:,:,1)./(real(obj.settings.Mint*obj.ucoeff_previous1(:,:,1)*2*pi)))/12 ...
                        +(obj.ucoeff_previous1(:,:,3)./(real(obj.settings.Mint*obj.ucoeff_previous1(:,:,3)*2*pi))...
                        - obj.ucoeff_previous1(:,:,2)./(real(obj.settings.Mint*obj.ucoeff_previous1(:,:,2)*2*pi)))*(2/3))/obj.dt;
                    if mod(obj.InvMeth,2)
                        fdt=gather(fdt);
                    end
                    [Vux,Vuz]=Linv_fdt(fdt,obj.Linv,obj.Msin,obj.mats.Mp1,obj.mats.Mp3,obj.settings,obj.zero_row,obj.settings.n*obj.settings.m/2+obj.settings.m/2+1);
                    obj.Transformed.Vux((i-2)/obj.saving_rate1,:) =gather(Vux);
                    obj.Transformed.Vuz((i-2)/obj.saving_rate1,:) =gather(Vuz);
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
                obj.ufull_save=gather(ucoeff);
            end
            if ( mod(i, obj.saving_rate2) == 2 )&& i~=2
                fdt_full_save=gather((-ucoeff./(real(obj.settings.Mint*ucoeff*2*pi))...
                    + obj.ucoeff_previous2(:,:,1)./(real(obj.settings.Mint*obj.ucoeff_previous2(:,:,1)*2*pi)))/12 ...
                    +(obj.ucoeff_previous2(:,:,3)./(real(obj.settings.Mint*obj.ucoeff_previous2(:,:,3)*2*pi))...
                    -obj.ucoeff_previous2(:,:,2)./(real(obj.settings.Mint*obj.ucoeff_previous2(:,:,2)*2*pi)))*(2/3))/obj.dt;
                udt_full_save=gather((-ucoeff...
                    + obj.ucoeff_previous2(:,:,1))/12 ...
                    +(obj.ucoeff_previous2(:,:,3)...
                    -obj.ucoeff_previous2(:,:,2))*(2/3))/obj.dt;
                ufull_save=obj.ufull_save; %#ok<PROPLC>
                t=(i-2)*obj.dt;
                save(['t' num2str(t) '.mat'],'t','ufull_save','fdt_full_save','udt_full_save');
            end
            if ( mod(i, obj.saving_rate2) == 1 )&& i~=1
                obj.ucoeff_previous2(:,:,3)=gather(ucoeff);
            end
            if ( mod(i, obj.saving_rate2) == obj.saving_rate2-1 )
                obj.ucoeff_previous2(:,:,2)=gather(ucoeff);
            end
            if ( mod(i, obj.saving_rate2) == obj.saving_rate2-2 )
                obj.ucoeff_previous2(:,:,1)=gather(ucoeff);
            end
        end
        
        function varargout=export(obj)
            if obj.InvMeth
                varargout{1}=obj.Transformed.ex;
                varargout{2}=obj.Transformed.ez;
                varargout{3}=obj.Transformed.ex_g;
                varargout{4}=obj.Transformed.ez_g;
                varargout{5}=obj.Transformed.Dxx;
                varargout{6}=obj.Transformed.Dxz;
                varargout{7}=obj.Transformed.Dzx;
                varargout{8}=obj.Transformed.Dzz;
                varargout{9}=obj.Transformed.Dxx_g;
                varargout{10}=obj.Transformed.Dxz_g;
                varargout{11}=obj.Transformed.Dzx_g;
                varargout{12}=obj.Transformed.Dzz_g;
                varargout{13}=obj.Transformed.Vix;
                varargout{14}=obj.Transformed.Viz;
                varargout{15}=obj.Transformed.Vix_g;
                varargout{16}=obj.Transformed.Viz_g;
                varargout{17}=obj.Transformed.VDTx;
                varargout{18}=obj.Transformed.VDTz;
                varargout{19}=obj.Transformed.VDTx_g;
                varargout{20}=obj.Transformed.VDTz_g;                
                
                if obj.varDir==1
                    varargout{13}=obj.Transformed.DDTxx;
                    varargout{14}=obj.Transformed.DDTzx;
                    varargout{15}=obj.Transformed.DDTxx_g;
                    varargout{16}=obj.Transformed.DDTzx_g;
                elseif obj.varDir==2
                    varargout{13}=obj.Transformed.DDTxz;
                    varargout{14}=obj.Transformed.DDTzz;
                    varargout{15}=obj.Transformed.DDTxx_g;
                    varargout{16}=obj.Transformed.DDTzx_g;
                end
                
                if obj.InvMeth>2 && nargout > 16
                    varargout{17}=obj.Transformed.Vux;
                    varargout{18}=obj.Transformed.Vuz;
                end
            else
                varargout = cell(1,nargout);
            end
        end
        function Transformed=export_struct(obj)
            if obj.InvMeth
                Transformed = obj.Transformed;
            else
                Transformed = struct([]);
            end
        end
    end
end

