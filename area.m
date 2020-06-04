function Sf=area(cf,theta)
    %% LEGACY S_e surface integral function (real)
    % Function to compute the Surface Area Integral of a field with FFT 
    % transform of the field.
    dtheta=theta(2)-theta(1);
    N=numel(theta);
    ind=real(cf(:,1)).*sin(theta);
    %% Closed Newton-Cote Equal Space Integration
    if mod(N,5)==1
        Sf=[19 repmat([75 50 50 75 38],1,(N-6)/5)  75 50 50 75 19]*ind*5/288*dtheta*pi*2;
    elseif mod(N,4)==1
        Sf=[7 repmat([32 12 32 14],1,(N-5)/4)  32 12 32 7]*ind*2/45*dtheta*pi*2;
    elseif mod(N,3)==1
        Sf=[1 repmat([3 3 2],1,(N-4)/3) 3 3 1]*ind*3/8*dtheta*pi*2;
    elseif mod(N,2)==1
        Sf=[1 repmat([4 2],1,(N-3)/2) 4 1]*ind/3*dtheta*pi*2;
    else
        Sf=trapz(ind)*dtheta*pi*2; %% Trapezoid Rule
    end
end