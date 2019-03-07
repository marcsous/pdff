function [params sse] = fresco(te,data,Tesla,varargin)
%[params sse] = fresco(te,data,Tesla,varargin)
%[params sse] = fresco(imDataParams,varargin)
%
% Field and phase regularized estimation using smooth
% constrained optimization of the proton density fat
% fraction (PDFF).
%
% Works best with phase unwrapping (e.g. unwrap2.m &
% unwrap3.m from https://github.com/marcsous/unwrap).
%
% Inputs:
%  te is echo time in seconds (vector)
%  data (1D, 2D or 3D with echos in last dimension)
%  Tesla field strength (scalar)
%  varargin in option/value pairs (e.g. 'ndb',2.5)
%
% Alternate input:
%  imDataParams is a struct for fat-water toolbox
%
% Outputs:
%  params.B0 is B0 (Hz)
%  params.R2 is R2* (1/s)
%  params.FF is PDFF (%)
%  params.PH is PH (rad)
%  sse is sum of squares error

% demo code
if nargin==0
    load PHANTOM_NDB_PAPER.mat
    varargin = {'h2o',4.8,'ndb',3};
elseif isa(te,'struct')
    % special handling for ISMRM F/W toolbox structure
    if nargin>1; varargin = {data,Tesla,varargin{:}}; end
    data = te.images*100/max(abs(te.images(:))); % no weirdness
    if te.PrecessionIsClockwise<0; data = conj(data); end
    Tesla = te.FieldStrength; % Tesla
    te = te.TE; % seconds
    [nx ny nz nc ne] = size(data);
    if nc>1; data = matched_filter(data); end
    data = reshape(data,nx,ny,nz,ne);
end

%% options

% defaults
opts.muB = 1e-3; % regularization for B0 (1e-3-1e-2)
opts.muR = 1e-3; % regularization for R2* (1e-3-1e-2)
opts.nonnegative = 0; % nonnegative water/fat (1=on 0=off)
opts.smooth_phase = 1; % smooth initial phase (1=on 0=off)
opts.smooth_field = 1; % smooth field (1=on 0=off)
opts.softplus = 1; % nonnegative R2* (1=on 0=off)
opts.filter = [1 3 1;3 9 3;1 3 1]>0; % low pass filter

% constants
opts.ndb = 2.5; % no. double bonds
opts.h2o = 4.7; % water frequency ppm
opts.maxit = 10; % max. no. iterations
opts.noise = []; % noise std (if available)

% debugging options
opts.unwrap = 1; % fieldmap phase unwrapping (0 for testing)
opts.psi = []; % fixed initial psi estimate ([] for testing)
opts.display = 1; % useful but slow (0 to turn off)
opts.iter = 0; % current iteration number (outer loop)

% varargin handling (must be option/value pairs)
for k = 1:2:numel(varargin)
    if k==numel(varargin) || ~ischar(varargin{k})
        error('''varargin'' must be option/value pairs.');
    end
    if ~isfield(opts,varargin{k})
        error('''%s'' is not a valid option.',varargin{k});
    end
    opts.(varargin{k}) = varargin{k+1};
end

%% argument checks - be flexible: 1D, 2D or 3D data

if numel(data)==numel(te)
    data = reshape(data,1,1,1,numel(te));
elseif ndims(data)==2 && size(data,2)==numel(te)
    data = permute(data,[1 3 4 2]);
elseif ndims(data)==3 && size(data,3)==numel(te)
    data = permute(data,[1 2 4 3]);
elseif ndims(data)==4 && size(data,4)==numel(te)
else
    error('data [%s\b] not compatible with te [%s\b].',...
    sprintf('%i ',size(data)),sprintf('%i ',size(te)));
end
if max(te)>1
    error('''te'' should be in seconds.');
end
if ~issorted(te)
    warning('''te'' should be sorted. Sorting.');
    [te k] = sort(te); data = data(:,:,:,k);
end
if isreal(data) || ~isfloat(data)
    error('data should be complex floats.');
end
[nx ny nz ne] = size(data);

%% see if gpu is possible 
try
    gpu = gpuDevice;
    if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b'); end
    data = gpuArray(data);
    fprintf(' GPU found = %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
catch ME
    data = gather(data);
    warning('%s. Using CPU.',ME.message);
end
%% setup

% estimate noise std dev
if isempty(opts.noise)
    X(:,9) = reshape(circshift(data,[0 0]),[],1);
    X(:,8) = reshape(circshift(data,[0 1]),[],1);
    X(:,7) = reshape(circshift(data,[1 0]),[],1);
    X(:,6) = reshape(circshift(data,[1 1]),[],1);
    X(:,5) = reshape(circshift(data,[2 0]),[],1);
    X(:,4) = reshape(circshift(data,[0 2]),[],1);
    X(:,3) = reshape(circshift(data,[2 1]),[],1);
    X(:,2) = reshape(circshift(data,[1 2]),[],1);
    X(:,1) = reshape(circshift(data,[2 2]),[],1);
    X = min(svd(X'*X)); % overwrite X (memory)
    opts.noise = gather(sqrt(X/nnz(data)));
end

% time evolution matrix
te = cast(te,'like',opts.noise);
[opts.A opts.psif] = fat_basis(te,Tesla,opts.ndb,opts.h2o);

% display
disp([' Data size: [' sprintf('%i ',size(data)) sprintf('\b]')])
disp([' TE (ms): ' sprintf('%.2f ',1000*te(:))])
disp([' Tesla (T): ' sprintf('%.2f',Tesla)])
disp(opts);

% standardize mu across datasets
opts.muR = opts.muR * opts.noise / Tesla;
opts.muB = opts.muB * opts.noise / Tesla;

%% center kspace (otherwise smoothing is risky)

mask = sqrt(real(dot(data,data,4))/ne);

data = fft3(data);
tmp = real(dot(data,data,4));
[~,k] = max(tmp(:));
[dx dy dz] = ind2sub([nx ny nz],k);
data = circshift(data,1-[dx dy dz]);
data = ifft3(data).*(mask>0);
fprintf(' Shifting kspace center from [%i %i %i]\n',dx,dy,dz);

%% allow faster calculations

% echos in 1st dim
te = reshape(te,ne,1);
data = permute(data,[4 1 2 3]);

% consistent data types
te = real(cast(te,'like',data));
opts.A = cast(opts.A,'like',data);
opts.filter = cast(opts.filter,'like',te);

%% initial estimates

if isempty(opts.psi)
    
    % initialize with dominant frequency (rad/s)
    tmp = dot(data(1:ne-1,:,:,:),data(2:ne,:,:,:),1);
    psi = angle(tmp)/min(diff(te))+i*imag(opts.psif); % rad/s
    
    % find the 2 local minima
    fprintf(' 1st local min\t');
    [psi1 r1] = nlsfit(psi,te,data,opts);

    fprintf(' 2nd local min\t');
    [psi2 r2] = nlsfit(psi1-opts.psif,te,data,opts);
    
    % choose the lowest ||r||
    ok = dot(r1,r1) < dot(r2,r2);
    psi(ok) = psi1(ok);
    psi(~ok) = psi2(~ok);

else
    
    % convert B0 to rad/s and same data type
    psi = 2*pi*real(opts.psi)+i*imag(opts.psi);
    if isscalar(psi)
        psi = repmat(cast(psi,'like',data),1,nx,ny,nz);
    else
        psi = reshape(cast(psi,'like',data),1,nx,ny,nz);
    end

end

%% main algorithm

t = tic;
for iter = 1:opts.maxit
    
    fprintf(' Outer loop %i\t',iter);
    
    % image processing to remove swaps
    if numel(te)~=numel(data)
        
        B0 = real(reshape(psi,size(mask)));

        if opts.unwrap
            B0 = B0*2*pi/real(opts.psif); % ±pi wraps
            if nz==1
                B0 = unwrap2(gather(B0),gather(mask>opts.noise));
            else
                B0 = unwrap3(gather(B0),gather(mask>opts.noise));
            end
            B0 = B0*real(opts.psif)/2/pi; % original wraps
            B0 = cast(B0,'like',te); % back on gpu (if used)
        end
        
        if opts.smooth_field
            B0 = convn(mask.*B0,opts.filter,'same');
            B0 = B0./max(convn(mask,opts.filter,'same'),opts.noise);
        end
 
        psi = reshape(B0,size(psi)) + i*imag(psi);
        
    end
    
    % local optimization (minimize residual)
    opts.iter = iter;
    [psi r phi x] = nlsfit(psi,te,data,opts);
    
end
toc(t);

%% return parameters in correct format

if opts.softplus; psi = softplus(psi); end
params.B0 = gather(squeeze(real(psi)/2/pi)); % B0 (Hz)
params.R2 = gather(squeeze(imag(psi))); % R2* (1/s)
params.FF = gather(squeeze(100*x(2,:,:,:)./sum(x))); % FF (%)
params.PH = gather(squeeze(phi)); % initial phase (rad)
sse = gather(squeeze(real(dot(r,r)))); % sum of squares error

%% nonlinear least squares fitting
function [psi r phi x] = nlsfit(psi,te,data,opts)

% regularizer (smooth B0 + zero R2*)
omega = real(psi);

% let the phase stabilize before smoothing it
if opts.iter<opts.maxit/2; opts.smooth_phase = 0; end

% Levenberg Marquardt damping term: use a high starting value
% to prevent solutions from jumping out of the local minima
lambda = repmat(100*opts.noise^2,size(psi));

for iter = 1:opts.maxit

    % residual and derivatives
    [r phi x JB JR] = pclsr(psi,te,data,opts);

    % Jacobian J = [JB JR] has 2x2 normal matrix J'*J such that
    % Re(J'*J) = [J1 J2;J2 J3] (phase constrained least squares)
    J1 = real(dot(JB,JB))+opts.muB^2+lambda;
    J2 = real(dot(JB,JR));
    J3 = real(dot(JR,JR))+opts.muR^2+lambda;
    
    % solution [Re(Δpsi) Im(Δpsi)] = -inv(Re(J'*J)) * Re(J'*r)
    % where inv(Re(J'*J)) = [J3 -J2;-J2 J1]/dtm (determinant)
    JBr = real(dot(JB,r))+opts.muB^2*real(psi-omega);
    JRr = real(dot(JR,r))+opts.muR^2*imag(psi-omega);
    dtm = J1.*J3-J2.^2;

    % update for psi
    dpsi = complex(J2.*JRr-J3.*JBr,J2.*JBr-J1.*JRr)./dtm;
    
    % keep updates that reduce ||r||
    s = pclsr(psi+dpsi,te,data,opts);
    ok = dot(s,s) < dot(r,r); 
    psi(ok) = psi(ok)+dpsi(ok);
    lambda(ok) = lambda(ok) / 10;
    lambda(~ok) = lambda(~ok) * 5;

end

% final phi and x
[r phi x] = pclsr(psi,te,data,opts);

% display (slow and ugly)
if numel(data)==numel(te)
    cplot(1e3*te,data,'o');hold on;cplot(1e3*te,data-r);hold off;xlabel('te (ms)');
    if opts.softplus; tmp = softplus(psi); else; tmp = psi; end
    txt = sprintf('||r||=%.1e B0=%.0f R2*=%.0f FF=%.1f',norm(r),real(tmp)/2/pi,imag(tmp),100*x(2)/sum(x));
    title(txt);drawnow;pause(0.2);
elseif opts.display
    mid = ceil(size(data,4)/2); % middle slice
    tmp = squeeze(psi(1,:,:,mid));
    if opts.softplus; tmp = softplus(tmp); end;
    tmp(:,:,2) = real(tmp)/2/pi; % B0 (Hz)
    tmp(:,:,4) = imag(tmp(:,:,1)); % R2* (1/s)
    tmp(:,:,3) = 100*x(2,:,:,mid)./sum(x(:,:,:,mid)); % FF (%)
    tmp(:,:,1) = phi(1,:,:,mid); % phase (rad)
    txt = {'\phi (rad)','B0 (Hz)','PDFF (%)','R2* (1/s)'};
    range = {[-1 1]*pi,[-1 1]*abs(opts.psif)/2,[-5 105],[0 0.5/min(te)]};
    for k = 1:4
        subplot(2,2,k); imagesc(real(tmp(:,:,k)),range{k});
        title(txt{k}); colorbar; axis off;
    end
    drawnow;
end
fprintf('\t||r||=%.4e\tμR=%.3e\tμB=%.3e\tmin(λ)=%.1e\tmax(λ)=%.1e\n',norm(r(:)),opts.muR,opts.muB,min(lambda(:)),max(lambda(:)));

% catch numerical instabilities (due to negative R2*)
if any(~isfinite(psi(:)))
    error('Problem is too ill-conditioned: use softplus or increase muR.'); 
end

%% phase constrained least squares residual r=W*A*x*exp(i*phi)-b
function [r phi x JB JR] = pclsr(psi,te,b,opts)

% note echos in dim 1
[ne nx ny nz] = size(b);
b = reshape(b,ne,nx*ny*nz);
psi = reshape(psi,1,nx*ny*nz);

% change of variable (softplus)
if opts.softplus; [psi dpsi] = softplus(psi); end

% complex fieldmap
W = exp(i*te*psi);

% M = Re(A'*W'*W*A) is a 2x2 matrix [M1 M2;M2 M3]
% with inverse [M3 -M2;-M2 M1]/dtm (determinant)
WW = real(W).^2+imag(W).^2;
M1 = real(conj(opts.A(:,1)).*opts.A(:,1)).' * WW;
M2 = real(conj(opts.A(:,1)).*opts.A(:,2)).' * WW;
M3 = real(conj(opts.A(:,2)).*opts.A(:,2)).' * WW;
dtm = M1.*M3-M2.^2;

% z = inv(M)*A'*W'*b
Wb = conj(W).*b;
z = bsxfun(@times,opts.A'*Wb,1./dtm);
z = [M3.*z(1,:)-M2.*z(2,:);M1.*z(2,:)-M2.*z(1,:)];

% p = z.'*M*z
p = z(1,:).*M1.*z(1,:) + z(2,:).*M3.*z(2,:)...
  + z(1,:).*M2.*z(2,:) + z(2,:).*M2.*z(1,:);

% initial phase (phi)
if opts.smooth_phase
    p = reshape(p,nx,ny,nz,1);
    p = convn(p,opts.filter,'same');
    p = reshape(p,1,nx*ny*nz);
end
phi = angle(p)/2; % -pi/2<phi<pi/2
x = real(bsxfun(@times,z,exp(-i*phi)));

% absorb sign of x into phi
x = bsxfun(@times,x,exp(i*phi));
phi = angle(sum(x)); % -pi<phi<pi
x = real(bsxfun(@times,z,exp(-i*phi)));

% nonnegative water and fat (0<=FF<=1)
if opts.nonnegative
    x = max(x,0);
    WAx = W.*(opts.A*x);
    a = dot(WAx,b)./max(dot(WAx,WAx),eps(opts.noise));
    x = bsxfun(@times,abs(a),x);
    phi = angle(a);
end

% residual
eiphi = exp(i*phi);
WAx = W.*(opts.A*x);
r = bsxfun(@times,WAx,eiphi);
r = reshape(r-b,ne,nx,ny,nz);

%% derivatives w.r.t. real(psi) and imag(psi)

% if not needed, return
if nargout<2; return; end

% y = inv(M)*A'*W'*T*b
y = bsxfun(@times,bsxfun(@times,opts.A,te)'*Wb,1./dtm);
y = [M3.*y(1,:)-M2.*y(2,:);M1.*y(2,:)-M2.*y(1,:)];

% q = y.'*M*z
q = y(1,:).*M1.*z(1,:) + y(2,:).*M3.*z(2,:)...
  + y(1,:).*M2.*z(2,:) + y(2,:).*M2.*z(1,:);

% H is like M but with T in the middle
WW = bsxfun(@times,te,WW);
H1 = real(conj(opts.A(:,1)).*opts.A(:,1)).' * WW;
H2 = real(conj(opts.A(:,1)).*opts.A(:,2)).' * WW;
H3 = real(conj(opts.A(:,2)).*opts.A(:,2)).' * WW;

% s = z.'*H*z
s = z(1,:).*H1.*z(1,:) + z(2,:).*H3.*z(2,:)...
  + z(1,:).*H2.*z(2,:) + z(2,:).*H2.*z(1,:); 

%% real part (B0): JB

% first term
JB = bsxfun(@times,i*te,WAx);

% second term
dphi = -real(q./p); dphi(p==0) = 0;
JB = JB + bsxfun(@times,WAx,i*dphi);

% third term
dx = y + bsxfun(@times,z,dphi);
dx = imag(bsxfun(@times,dx,1./eiphi));
JB = JB + W.*(opts.A*dx);

% impart phase
JB = bsxfun(@times,JB,eiphi);

%% imag part (R2*): JR

% first term
JR = bsxfun(@times,-te,WAx);

% second term
dphi = -imag(q./p)+imag(s./p); dphi(p==0) = 0;
JR = JR + bsxfun(@times,WAx,i*dphi);

% third term
dx = y + bsxfun(@times,z,i*dphi);
dx = real(bsxfun(@times,dx,-1./eiphi));

Hx = [H1.*x(1,:)+H2.*x(2,:);H3.*x(2,:)+H2.*x(1,:)];
Hx = bsxfun(@times,Hx,2./dtm);
dx = dx+[M3.*Hx(1,:)-M2.*Hx(2,:);M1.*Hx(2,:)-M2.*Hx(1,:)];

JR = JR + W.*(opts.A*dx);

% impart phase
JR = bsxfun(@times,JR,eiphi);

% change of variable (softplus)
if opts.softplus; JR = bsxfun(@times,JR,dpsi); end

%% return arguments

x = reshape(x,2,nx,ny,nz);
JB = reshape(JB,ne,nx,ny,nz);
JR = reshape(JR,ne,nx,ny,nz);
phi = reshape(phi,1,nx,ny,nz);

%% softplus function: x=log(1+exp(x))
function [psi dpsi] = softplus(psi)

% only change imag part (R2)
B0 = real(psi);
R2 = imag(psi);

% log(exp) loses precision so use double
ok = R2 < log(flintmax('single'));
expR2 = exp(double(R2(ok)));
R2(ok) = log(1+expR2);

% back to complex
psi = complex(B0,R2);

% derivative wrt R2
dpsi = ones(size(R2),'like',R2);
dpsi(ok) = expR2./(1+expR2);
