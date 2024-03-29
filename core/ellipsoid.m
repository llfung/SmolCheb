function [B,Vmin,Vmax,M,A1,A2,A3]=ellipsoid(AR)
    B=(AR.^2-1)./(AR.^2+1); 
    if nargout>1
        % Kim's version - more prone to numerical problem at high AR
%                 e=sqrt(1-1./AR.^2);
%                 L=log((1+e)./(1-e));
%                  Vmax=1./(8/3*e.^3./(-2*e+(1+e.^2).*L))./AR;
%                  Vmin=1./(16/3*e.^3./(2*e+(3*e.^2-1).*L))./AR;
                
        % Modified Cabrera et al. (2021) Appendix A version
        alpha=AR.^2./(AR.^2-1)-AR.*acosh(AR).*(sqrt(AR.^2-1).^(-3));
        chi=2*AR.*acosh(AR)./sqrt(AR.^2-1);
        gamma=-2./(AR.^2-1)+2*AR.*acosh(AR).*(sqrt(AR.^2-1).^(-3));
        Vmax=(chi+AR.^2.*gamma)*3/8./AR;
        Vmin=(chi+alpha)*3/8./AR;
    end
    if nargout>3
        e=sqrt(1-1./AR.^2);
        F_fac=pi*e.^2./(315*((e.^2+1).*atanh(e)-e).^2.*((1-3*e.^2).*atanh(e)-e));
        F=(-(420*e+2240*e.^3+4249*e.^5-2152*e.^7)...
            +(420+3360*e.^2+1890*e.^4-1470*e.^6).*atanh(e)...
            -(1260*e-1995*e.^3+2730*e.^5-1995*e.^7).*atanh(e).^2).*F_fac;
        
        M=3/16/pi*AR.^4./(AR.^4-1).*(-1+(-1+2*AR.^2)./(AR.*sqrt(-1+AR.^2))...
            .*log(AR+sqrt(-1+AR.^2))).*F;

    end
    if nargout>4
        xi=1/sqrt(1 - AR^-2);
    	A1 = -16*pi / ( 9*xi^3* (-3*xi + acoth(xi)*(-1 + 3*xi^2)) );
    	A2 =  16*pi * (-1 + xi^2) / (3*xi^2 *(-1 + 2*xi^2) * (2 - 3*xi^2 + (3*acoth(xi)) *xi *(-1 + xi^2)));
    	A3 = -32*pi * (-1 + xi^2) / (3*xi^3 *(5*xi - 3*xi^3 + (3*acoth(xi)) *(-1 + xi^2)^2));
    end
end