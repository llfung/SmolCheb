classdef chebyshev
    properties
        col_pt %Chebyshev Collocation Points
        DM %Chebyshev Differential Matrices (3rd dimenion is the order of differentiation)
        N uint16 %Size of Matrices and Collocation Points Vector
        N_original uint16 %Original number of collocation points before boundary application
        M uint8 %Max Order of Chebyshev Differential Matrices
        bc_opt bc_type %Boundary Type
        tran_opt tran_type %Transformation Type
        pix %y-domain defintion by changing the numerical pi value.
        wint %Integration weight vector
    end
    methods
        function D_out=D(obj,M)
            D_out=obj.DM(:,:,M);
        end
        function obj=chebyshev(siz,M,bc_opt,tran_opt,pix)
            narginchk(1,5)
            obj.N=siz;
            obj.N_original=siz;
            if nargin < 2
                M=4;
            end
            if nargin < 5
                obj.pix=0.99*pi;
            else
                obj.pix=pix;
            end
            [obj.col_pt,obj.DM]=chebdif(obj.N,M);
            obj.M=M;
            if nargin < 3
                obj.bc_opt=bc_type.none;
            else
                obj=bc(obj,bc_opt);
            end
            if nargin < 4
                obj.tran_opt=tran_type.none;
                obj.wint=Chebyweight_infun(siz);
            else
                obj=tran(obj,tran_opt);
                if tran_opt==tran_type.lin
                    obj.wint=Chebyweight_infun(siz)./2;
                end
                if tran_opt==tran_type.none
                    obj.wint=Chebyweight_infun(siz);
                end
            end
            
                
        end
        function obj=bc(obj,bc_opt)
            if nargin < 2
                if obj.M > 3
                    bc_opt=bc_type.b4c;
                else
                    bc_opt=bc_type.dirichlet;
                end
            end
            switch bc_opt
                case bc_type.none
                    obj.N=obj.N_original;
                    [obj.col_pt,obj.DM]=chebdif(obj.N,obj.M);
                case bc_type.b4c
                    if obj.N~=obj.N_original
                        [obj.col_pt,obj.DM]=chebdif(obj.N_original,obj.M);
                    end
                    obj.DM=obj.DM(2:end-1,2:end-1,:);
                    obj.col_pt=obj.col_pt(2:end-1);
                    obj.DM(:,:,4)=cheb4c(obj.N_original);
                    obj.N=obj.N_original-2;
                case bc_type.dirichlet
                    if obj.N~=obj.N_original
                        [obj.col_pt,obj.DM]=chebdif(obj.N_original,obj.M);
                    end
                    obj.DM=obj.DM(2:end-1,2:end-1,:);
                    obj.col_pt=obj.col_pt(2:end-1);
                    obj.N=obj.N_original-2;
                %case bc_type.robin %Not yet implemented
                case bc_type.dirichlet_up
                    if obj.N~=obj.N_original
                        [obj.col_pt,obj.DM]=chebdif(obj.N_original,obj.M);
                    end
                    obj.DM=obj.DM(2:end,2:end,:);
                    obj.col_pt=obj.col_pt(2:end);
                    obj.N=obj.N_original-1;
               case bc_type.dirichlet_low
                    if obj.N~=obj.N_original
                        [obj.col_pt,obj.DM]=chebdif(obj.N_original,obj.M);
                    end
                    obj.DM=obj.DM(1:end-1,1:end-1,:);
                    obj.col_pt=obj.col_pt(1:end-1);
                    obj.N=obj.N_original-1;
%                 case clamped_up_dirichlet_low
%                     if obj.N~=obj.N_original
%                         [obj.col_pt,obj.DM]=chebdif(obj.N_original,obj.M);
%                     end
%                     obj.DM=obj.DM(1:end-1,1:end-1,:);
%                     obj.col_pt=obj.col_pt(1:end-1);
%                     obj.N=obj.N_original-1;
                otherwise
                    error('Unknown Boundary Condition Type');
            end
            obj.bc_opt=bc_opt;
        end
        function integ=cheb_int(obj,v)
            switch obj.tran_opt
                case tran_type.none
                    switch obj.bc_opt
                        case bc_type.none
                        case {bc_type.b4c,bc_type.dirichlet}
                            v = [0;v;0];
                        otherwise
                            error('Unknown Boundary Condition Type');
                    end
                case tran_type.tan
                    pix=obj.pix;
                    switch obj.bc_opt
                        case bc_type.none
                        case {bc_type.b4c,bc_type.dirichlet}
                            v = [0;v;0];
                            y=[tan(pix/2);obj.col_pt;tan(-pix/2)];
                        otherwise
                            error('Unknown Boundary Condition Type');
                    end
                    
                    dy=2/pix./(1+y.^2); %d(xi)/dy
                    v=v./dy;
                case tran_type.tansc
                    pix=obj.pix;
                    switch obj.bc_opt
                        case bc_type.none
                            y=obj.col_pt;
                        case {bc_type.b4c,bc_type.dirichlet}
                            v = [0;v;0];
                            y=[1;obj.col_pt;-1];
                        otherwise
                            error('Unknown Boundary Condition Type');
                    end
                    
                    dy=2*tan(pix/2)/pix./(1+(y/tan(pix/2)).^2); %d(xi)/dy
                    v=v./dy;
                case tran_type.sec
                    pix=obj.pix;
                    [xi,~]=chebdif(obj.N_original,1);
                    switch obj.bc_opt
                        case bc_type.none
                        case {bc_type.b4c,bc_type.dirichlet}
                            v = [0;v;0];
                            %y=[1/cos(pix/2);obj.col_pt;-1/cos(-pix/2)];
                        otherwise
                            error('Unknown Boundary Condition Type');
                    end
                    dy=1./(1./cos(pix/2.*xi)+pix/2.*xi./cos(pix/2.*xi).*tan(pix/2.*xi)); %d(xi)/dy
                    v=v./dy;
                case tran_type.lin
                    switch obj.bc_opt
                        case bc_type.none
                        case {bc_type.b4c,bc_type.dirichlet}
                            v = [0;v;0];
                        otherwise
                            error('Unknown Boundary Condition Type');
                    end
                    v=v./2;
                otherwise
                    error('Unknown Transformation Type');
            end
            N = length(v) - 1; %#ok<*PROPLC>
            v = [v; flipud(v(2:N))];

            a = fft(v)/N;
            a = [a(1)/2; a(2:N); a(N+1)/2];
            
            integ=0;
            for l=1:2:length(a)
                integ=integ+2*a(l)/(1-(l-1)^2);
            end
            %integ=real(integ);
        end
        function obj=tran(obj,tran_opt)
            if nargin < 2
                    tran_opt=tran_type.tan;
            end
            switch tran_opt
                case tran_type.none
                case tran_type.tan
                    pix=obj.pix;
                    %% Transformation dy
                    y=tan(pix/2.*obj.col_pt);
                    dy=2/pix./(1+y.^2); %d(xi)/dy
                    ddy=-4/pix.*y./((1+y.^2).^2);  %d2(xi)/dy2
                    dddy=4/pix.*(3.*y.^2-1)./(1+y.^2).^3; %d3(xi)/dy3
                    ddddy=-48/pix.*y.*(y.^2-1)./(1+y.^2).^4;%d4(xi)/dy4
                case tran_type.tansc
                    pix=obj.pix;
                    sc=tan(pix/2);
                    %% Transformation dy
                    ysc=tan(pix/2.*obj.col_pt);
                    y=ysc/sc;
                    dy=2*sc/pix./(1+ysc.^2); %d(xi)/dy
                    ddy=-4*sc^2/pix.*ysc./((1+ysc.^2).^2);  %d2(xi)/dy2
                    dddy=4*sc^3/pix.*(3.*ysc.^2-1)./(1+ysc.^2).^3; %d3(xi)/dy3
                    ddddy=-48*sc^4/pix.*ysc.*(ysc.^2-1)./(1+ysc.^2).^4;%d4(xi)/dy4
                case tran_type.sec
                    %% Transformation dy
                    xi=obj.col_pt;
                    pix=obj.pix;
                    y=xi./cos(pix/2.*xi);
                    dy=1./(1./cos(pix/2.*xi)+pix/2.*xi./cos(pix/2.*xi).*tan(pix/2.*xi)); %d(xi)/dy
                    ddy=(pi.*(-3.*pi.*xi + pi.*xi.*cos(pi.*xi) - 4.*sin(pi.*xi)))./(2 + pi.*xi.*tan((pi.*xi)./2)).^3;  %d2(xi)/dy2
                    dddy=(pi.^2./cos((pi.*xi)./2).*(-28 + 15.*pi.^2.*xi.^2 ...
                    - 48.*cos(pi.*xi) +(-20 + pi.^2.*xi.^2).*cos(2.*pi.*xi)+...
                        16.*pi.*xi.*sin(pi.*xi) - 8.*pi.*xi.*sin(2.*pi.*xi)))./ ...
                        (4.*(2 + pi.*xi.*tan((pi.*xi)./2)).^5)+...
                        (pi.*cos((pi.*xi)./2).*(-3.*pi.*xi + pi.*xi.*cos(pi.*xi)...
                        - 4.*sin(pi.*xi)))./(2.*(2.*cos((pi.*xi)./2) + pi.*xi.*sin((pi.*xi)./2)).^2)...
                        .*ddy; %d3(xi)/dy3
                    ddddy=((pi.^3./cos((pi.*xi)./2).^3.*(268.*pi.*xi - 41.*pi.^3.*xi.^3 - 8.*pi.*xi.*(-26 + pi.^2.*xi.^2).*cos(pi.*xi) + pi.*xi.*(-60 + pi.^2.*xi.^2).*cos(2.*pi.*xi) + 256.*sin(pi.*xi) + 40.*pi.^2.*xi.^2.*sin(pi.*xi) + 128.*sin(2.*pi.*xi) - 12.*pi.^2.*xi.^2.*sin(2.*pi.*xi)))./(32.*(2 + pi.*xi.*tan((pi.*xi)./2)).^4)).*dy.^3 ...
                        +(((pi.^2./cos((pi.*xi)./2).^3.*(-28 + 15.*pi.^2.*xi.^2 - 48.*cos(pi.*xi) + (-20 + pi.^2.*xi.^2).*cos(2.*pi.*xi) + 16.*pi.*xi.*sin(pi.*xi) - 8.*pi.*xi.*sin(2.*pi.*xi)))./(16.*(2 + pi.*xi.*tan((pi.*xi)./2)).^3))).*3.*dy.*ddy+ ...
                        +(ddy.*ddy); %d4(xi)/dy4

                case tran_type.lin
                    y=obj.col_pt./2+0.5;
                    one_col=ones(size(obj.col_pt));
                    dy=2*(one_col);
                    ddy=0.*(one_col);
                    dddy=0.*(one_col);
                    ddddy=0.*(one_col);
                otherwise
                    error('Unknown Transformation Type');
            end
            if tran_opt~=tran_type.none
                    %% Transformation applied onto Chevyshev differential matrices
                    if obj.M>0
                        D1T=diag(dy)*obj.D(1);
                    end
                    if obj.M>1
                        D2T=diag(dy.^2)*obj.D(2)+diag(ddy)*obj.D(1);
                    end
                    if obj.M>2
                        D3T=diag(dy.^3)*obj.D(3)+3.*diag(dy.*ddy)*obj.D(2)+diag(dddy)*obj.D(1);
                    end
                    if obj.M>3
                        D4T=diag(dy.^4)*obj.D(4)+6.*diag(ddy.*dy.^2)*obj.D(3)+diag(3.*ddy.^2+4.*dy.*dddy)*obj.D(2)+diag(ddddy)*obj.D(1);
                    end
                    if obj.M>0
                        obj.DM(:,:,1)=D1T;
                    end
                    if obj.M>1
                        obj.DM(:,:,2)=D2T;
                    end
                    if obj.M>2
                        obj.DM(:,:,3)=D3T;
                    end
                    if obj.M>3
                        obj.DM(:,:,4)=D4T;
                    end
                    if obj.M>4
                        error('Transformation Order Exceed Support');
                    end
                    obj.col_pt=y;
            end
            
            obj.tran_opt=tran_opt;
        end
        function I_out=I(obj)
            I_out=eye(obj.N);
        end
        function O_Out=O(obj)
            O_Out=zeros(obj.N,obj.N);
        end
    end
end

    
    

function D4 = cheb4c(N)

%  The function D4 =  cheb4c(N) computes the fourth 
%  derivative matrix on Chebyshev interior points, incorporating 
%  the clamped boundary conditions u(1)=u'(1)=u(-1)=u'(-1)=0.
%
%  Input:
%  N:     N-2 = Order of differentiation matrix.  
%               (The interpolant has degree N+1.)
%
%  Output:
%  x:      Interior Chebyshev points (vector of length N-2)
%  D4:     Fourth derivative matrix  (size (N-2)x(N-2))
%
%  The code implements two strategies for enhanced 
%  accuracy suggested by W. Don and S. Solomonoff in 
%  SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
%  The two strategies are (a) the use of trigonometric 
%  identities to avoid the computation of differences 
%  x(k)-x(j) and (b) the use of the "flipping trick"
%  which is necessary since sin t can be computed to high
%  relative precision when t is small whereas sin (pi-t) cannot.
   
%  J.A.C. Weideman, S.C. Reddy 1998.
    N=double(N);
    I = eye(N-2);                   % Identity matrix.
    L = logical(I);                 % Logical identity matrix.

   n1 = floor(N/2-1);               % n1, n2 are indices used 
   n2 = ceil(N/2-1);                % for the flipping trick.

    k = [1:N-2]';                   % Compute theta vector.
   th = k*pi/(N-1);                 

    x = sin(pi*[N-3:-2:3-N]'/(2*(N-1))); % Compute interior Chebyshev points.

    s = [sin(th(1:n1)); flipud(sin(th(1:n2)))];   % s = sin(theta)
                               
alpha = s.^4;                       % Compute weight function
beta1 = -4*s.^2.*x./alpha;          % and its derivatives.
beta2 =  4*(3*x.^2-1)./alpha;   
beta3 = 24*x./alpha;
beta4 = 24./alpha;
    B = [beta1'; beta2'; beta3'; beta4'];

    T = repmat(th/2,1,N-2);                
   DX = 2*sin(T'+T).*sin(T'-T);     % Trigonometric identity 
   DX = [DX(1:n1,:); -flipud(fliplr(DX(1:n2,:)))];   % Flipping trick. 
DX(L) = ones(N-2,1);                % Put 1's on the main diagonal of DX.

   ss = s.^2.*(-1).^k;              % Compute the matrix with entries
    S = ss(:,ones(1,N-2));          % c(k)/c(j)
    C = S./S';                      

    Z = 1./DX;                      % Z contains entries 1/(x(k)-x(j)).
 Z(L) = zeros(size(x));             % with zeros on the diagonal.

    X = Z';                         % X is same as Z', but with 
 X(L) = [];                         % diagonal entries removed.
    X = reshape(X,N-3,N-2);

    Y = ones(N-3,N-2);              % Initialize Y and D vectors.
    D = eye(N-2);                   % Y contains matrix of cumulative sums,
                                    % D scaled differentiation matrices.
    for ell = 1:4
              Y = cumsum([B(ell,:); ell*Y(1:N-3,:).*X]); % Recursion for diagonals
              D = ell*Z.*(C.*repmat(diag(D),1,N-2)-D);   % Off-diagonal
           D(L) = Y(N-2,:);                              % Correct the diagonal
    DM(:,:,ell) = D;                                     % Store D in DM
    end

   D4 = DM(:,:,4);                  % Extract fourth derivative matrix
end


function [x, DM] = chebdif(N, M)

%  The function [x, DM] =  chebdif(N,M) computes the differentiation 
%  matrices D1, D2, ..., DM on Chebyshev nodes. 
% 
%  Input:
%  N:        Size of differentiation matrix.        
%  M:        Number of derivatives required (integer).
%  Note:     0 < M <= N-1.
%
%  Output:
%  DM:       DM(1:N,1:N,ell) contains ell-th derivative matrix, ell=1..M.
%
%  The code implements two strategies for enhanced 
%  accuracy suggested by W. Don and S. Solomonoff in 
%  SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
%  The two strategies are (a) the use of trigonometric 
%  identities to avoid the computation of differences 
%  x(k)-x(j) and (b) the use of the "flipping trick"
%  which is necessary since sin t can be computed to high
%  relative precision when t is small whereas sin (pi-t) cannot.
%  Note added May 2003:  It may, in fact, be slightly better not to
%  implement the strategies (a) and (b).   Please consult the following
%  paper for details:   "Spectral Differencing with a Twist", by
%  R. Baltensperger and M.R. Trummer, to appear in SIAM J. Sci. Comp. 

%  J.A.C. Weideman, S.C. Reddy 1998.  Help notes modified by 
%  JACW, May 2003.
    N=double(N);
    M=double(M);
     I = eye(N);                          % Identity matrix.     
     L = logical(I);                      % Logical identity matrix.

    n1 = floor(N/2); n2  = ceil(N/2);     % Indices used for flipping trick.

     k = [0:N-1]';                        % Compute theta vector.
    th = k*pi/(N-1);

     x = sin(pi*[N-1:-2:1-N]'/(2*(N-1))); % Compute Chebyshev points.

     T = repmat(th/2,1,N);                
    DX = 2*sin(T'+T).*sin(T'-T);          % Trigonometric identity. 
    DX = [DX(1:n1,:); -flipud(fliplr(DX(1:n2,:)))];   % Flipping trick. 
 DX(L) = ones(N,1);                       % Put 1's on the main diagonal of DX.

     C = toeplitz((-1).^k);               % C is the matrix with 
C(1,:) = C(1,:)*2; C(N,:) = C(N,:)*2;     % entries c(k)/c(j)
C(:,1) = C(:,1)/2; C(:,N) = C(:,N)/2;

     Z = 1./DX;                           % Z contains entries 1/(x(k)-x(j))  
  Z(L) = zeros(N,1);                      % with zeros on the diagonal.

     D = eye(N);                          % D contains diff. matrices.
                                          
    for ell = 1:M
              D = ell*Z.*(C.*repmat(diag(D),1,N) - D); % Off-diagonals
           D(L) = -sum(D');                            % Correct main diagonal of D
    DM(:,:,ell) = D;                                   % Store current D in DM
    end
end
function [wint]=Chebyweight_infun(N)
%% Chebyshev Collocation Integrating weight factors
    wint=zeros(N,1);
    for iy=0:N-1
      wint(iy+1)=-.5;
        for ni=0:(N+1)/2-3
           wint(iy+1) = wint(iy+1) + cos(pi*(2*(ni+1)*iy)/(N-1))/((2*(ni+1))^2-1);   
        end
         wint(iy+1) = wint(iy+1) + .5*cos(pi*iy)/((N-1)^2-1);
         wint(iy+1) = -4./(N-1)*wint(iy+1);
        if  (iy == 0  || iy == (N-1)) 
            wint(iy+1)=wint(iy+1)*.5;
        end
    end
end