function [params sse] = presco(te,data,Tesla,varargin)
%[params sse] = presco(te,data,Tesla,varargin)
%[params sse] = presco(imDataParams,varargin)
%
% Phase regularized estimation using smoothing and
% constrained optimization for the proton density
% fat fraction (PDFF).
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
    load invivo.mat;
    %load PHANTOM_NDB_PAPER.mat
    %load phantom_16echo_bipolar.mat;
    %te = te(1:2:end); data=data(:,:,1:2:end);
    %load liver_gre_3d_2x3.mat; data = data(:,:,31,:);
    %load liver_gre_3d_1x6.mat; data = data(:,:,24,:); 
elseif isa(te,'struct')
    % partial handling of ISMRM F/W toolbox structure
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
opts.muB = 0.00; % regularization for B0
opts.muR = 0.01; % regularization for R2*
opts.nonnegFF = 0; % nonnegative pdff (1=on 0=off)
opts.nonnegR2 = 1; % nonnegative R2* (1=on 0=off)
opts.smooth_phase = 1; % smooth phase (1=on 0=off)
opts.smooth_field = 1; % smooth field (1=on 0=off)

% constants
opts.ndb = 2.5; % no. double bonds
opts.h2o = 4.7; % water frequency ppm
opts.filter = ones(3); % low pass filter
opts.maxit = [20 10]; % max. iterations (inner outer)
opts.noise = []; % noise std (if available)

% debugging options
opts.unwrap = 1; % use phase unwrapping of fieldmap (0=off)
opts.psi = []; % fixed initial psi estimate (for testing)
opts.display = 1; % useful but slow (0 to turn off)
opts.none = 0; % quick flag turn off all constraints

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
if opts.none
    opts.muB = 0;
    opts.muR = 0;
    opts.nonnegFF = 0;
    opts.nonnegR2 = 0;
    opts.smooth_phase = 0;
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

%% setup

% estimate noise std dev (smallest sing. value)
if isempty(opts.noise)
    tmp = abs(data); % remove field variations
    X(:,9) = reshape(circshift(tmp,[0 0]),[],1);
    X(:,8) = reshape(circshift(tmp,[0 1]),[],1);
    X(:,7) = reshape(circshift(tmp,[1 0]),[],1);
    X(:,6) = reshape(circshift(tmp,[1 1]),[],1);
    X(:,5) = reshape(circshift(tmp,[2 0]),[],1);
    X(:,4) = reshape(circshift(tmp,[0 2]),[],1);
    X(:,3) = reshape(circshift(tmp,[2 1]),[],1);
    X(:,2) = reshape(circshift(tmp,[1 2]),[],1);
    X(:,1) = reshape(circshift(tmp,[2 2]),[],1);
    X = sqrt(2) * min(svd(X'*X));
    opts.noise = gather(sqrt(X/nnz(tmp)));
end

% time evolution matrix
te = real(cast(te,'like',data));
[opts.A opts.psif] = fat_basis(te,Tesla,opts.ndb,opts.h2o);

% display
disp([' Data size: [' sprintf('%i ',size(data)) sprintf('\b]')])
disp([' TE (ms): ' sprintf('%.2f ',1000*te(:))])
disp([' Tesla (T): ' sprintf('%.2f',Tesla)])
disp(opts);

% standardize mu across datasets
if opts.noise
    opts.muR = opts.muR * opts.noise;
    opts.muB = opts.muB * opts.noise;
end

%% center kspace (otherwise smoothing is risky)

mask = dot(data,data,4);

data = fft3(data);
tmp = dot(data,data,4);
[~,k] = max(tmp(:));
[dx dy dz] = ind2sub([nx ny nz],k);
data = circshift(data,1-[dx dy dz]);
data = ifft3(data).*(mask>0);
fprintf(' Shifting kspace center from [%i %i %i]\n',dx,dy,dz);

opts.mask = mask > opts.noise^2;

%% initial estimates

if isempty(opts.psi)

    % dominant frequency (rad/s)
    tmp = dot(data(:,:,:,1:ne-1),data(:,:,:,2:ne),4);
    psi = angle(tmp)/min(diff(te))+i*imag(opts.psif);

else
    
    % supplied psi (convert to rad/s)
    psi = 2*pi*real(opts.psi)+i*imag(opts.psi);
    
    if isscalar(psi)
        psi = repmat(psi,nx,ny,nz);
    else
        psi = reshape(psi,nx,ny,nz);
    end

end

psi(~opts.mask) = 0;

%% faster calculations

% check for gpu
try
    gpu = gpuDevice;
    if verLessThan('matlab','8.4'); error('GPU needs MATLAB R2014b'); end
    data = gpuArray(data);
    fprintf(' GPU found = %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
catch ME
    data = gather(data);
    warning('%s. Using CPU.',ME.message);
end

% echos in 1st dim
te = reshape(te,ne,1);
psi = permute(psi,[4 1 2 3]);
data = permute(data,[4 1 2 3]);

% consistent data types
psi = cast(psi,'like',data);
te = real(cast(te,'like',data));
opts.A = cast(opts.A,'like',data);
opts.muR = cast(opts.muR,'like',te);
opts.muB = cast(opts.muB,'like',te);
opts.noise = cast(opts.noise,'like',te);
opts.filter = cast(opts.filter,'like',te);

% allow for spatially varying regularization
if ~isscalar(opts.muR)
    opts.muR = reshape(opts.muR,size(psi));
end
if ~isscalar(opts.muB)
    opts.muB = reshape(opts.muB,size(psi));
end

%% main algorithm

% the first opts.maxit-1 iterations are really only to get
% rid of phase swaps. some constraints don't make sense if
% there are still fat-water swaps present (e.g. muB) so we
% could take liberties and apply them only when they make
% sense. in principle this might mean setting muB to 0 except
% on the last few iterations although in practice it doesn't
% seem to matter very much. using hernando or berglund B0
% initial estimates would be another way to remove swaps.
% the main goal here is to improve the solution close to
% the local minimum

for iter = 1:opts.maxit(end)
    
    fprintf(' Outer loop %i\n',iter);

    % resolve phase swaps
    if iter>1 && numel(te)~=numel(data) 
        
        % global swaps
        if opts.unwrap
            r = pclsr(psi,te,data,opts);
            s = pclsr(psi-opts.psif,te,data,opts);
            if norm(s(:)) < norm(r(:))
                psi(opts.mask) = psi(opts.mask) - opts.psif;
            end
        end

        % regional swaps
        if opts.unwrap
            psi = unswap(psi,te,data,opts);
        end
       
        % pixel swaps (smooth field)
        if opts.smooth_field
            B0 = real(squeeze(psi)); % only smooth B0
            B0 = medfiltn(B0,size(opts.filter),opts.mask);
            psi = reshape(B0,size(psi))+i*imag(psi);
        end
        
    end
    
    % fiddle with options on early iterations
    tmp = opts;
    if iter < opts.maxit(end)/2
        tmp.smooth_phase = 0;
        tmp.smooth_field = 0;
        tmp.nonnegR2 = 1;
        tmp.muB = 0;
        tmp.muR = opts.noise;        
    end
    
    % local optimization
    [r psi phi x] = nlsfit(psi,te,data,tmp);
    
end

%% return parameters in correct format

psi = transform(psi,opts);
params.B0 = gather(squeeze(real(psi)))/2/pi; % B0 (Hz)
params.R2 = gather(squeeze(imag(psi))); % R2* (1/s)
params.FF = gather(squeeze(100*x(2,:,:,:)./sum(x))); % FF (%)
params.FF(~opts.mask) = 0; % remove NaNs from 0/0 pixels
params.PH = gather(squeeze(phi)); % initial phase (rad)
sse = gather(squeeze(real(dot(r,r)))); % sum of squares error

%% nonlinear least squares fitting
function [r psi phi x] = nlsfit(psi,te,data,opts)

% regularizer (smooth B0 + zero R2)
PSI = real(psi);

for iter = 1:opts.maxit(1)

    % residual and Jacobian J = [JB JR]
    [r phi x JB JR] = pclsr(psi,te,data,opts);
    
    % gradient G = J'*r = [gB gR]
    gB = real(dot(JB,r))+opts.muB.^2.*real(psi-PSI);
    gR = real(dot(JR,r))+opts.muR.^2.*imag(psi-PSI);
    G = complex(gB,gR);

    % Hessian H ~ Re(J'*J) = [H1 H2; H2 H3]
    H1 = real(dot(JB,JB))+opts.muB.^2;
    H2 = real(dot(JB,JR));
    H3 = real(dot(JR,JR))+opts.muR.^2;

    % Cauchy point = (G'G)/(G'HG)
    GG = gB.^2+gR.^2;
    GHG = gB.^2.*H1+2*gB.*gR.*H2+gR.^2.*H3;

    % damping to reduce step size
    damp = median(GHG(opts.mask));
    step = GG ./ (GHG+damp/iter);
    dpsi = -step.*G;
    
    % cost function = sum of squares residual + penalty terms
    cost = @(arg)sum(abs(pclsr(arg,te,data,opts)).^2)+...
           opts.muB.^2.*real(transform(arg,opts)-PSI).^2+...
           opts.muR.^2.*imag(transform(arg,opts)-PSI).^2; 

    % crude linesearch
    for k = 1:3
        ok = cost(psi+dpsi) < cost(psi);
        psi(ok) = psi(ok) + dpsi(ok);
        dpsi(~ok) = dpsi(~ok) / 2;
    end

end

% final phi and x
[r phi x] = pclsr(psi,te,data,opts);

% display (slow and ugly)
tmp = transform(psi,opts);
if numel(data)==numel(te)
    cplot(1e3*te,data,'o');hold on;cplot(1e3*te,data-r);hold off;xlabel('te (ms)');
    txt = sprintf('||r||=%.2e B0=%.1f R2*=%.1f FF=%.1f',norm(r),real(tmp)/2/pi,imag(tmp),100*x(2)/sum(x));
    title(txt);drawnow;pause(0.1);
elseif opts.display
    mid = ceil(size(data,4)/2); % middle slice
    tmp(1,:,:,4) = abs(imag(tmp(1,:,:,mid))); % R2* (1/s)
    tmp(1,:,:,3) = abs(100*x(2,:,:,mid)./sum(x(:,:,:,mid))); % FF (%)
    tmp(1,:,:,2) = real(tmp(1,:,:,mid))/2/pi; % B0 (Hz)
    tmp(1,:,:,1) = phi(1,:,:,mid); % phase (rad)
    tmp = real(tmp); tmp = squeeze(tmp); tmp(isnan(tmp)) = 0; % NaN from 0/0
    txt = {'\phi (rad)','B0 (Hz)','PDFF (%)','R2* (1/s)'};
    s = {[-1 1]*pi,[-1 1]*abs(opts.psif)/7,[0 109],[0 0.2/min(te)]};
    for k = 1:4
        subplot(2,2,k); imagesc(tmp(:,:,k),s{k});
        title(txt{k}); colorbar; axis off;
    end
    drawnow;
end 

% catch numerical instabilities (due to negative R2*)
if any(~isfinite(psi(:)))
    error('Problem is too ill-conditioned: use nonnegR2 or increase muR.'); 
end

%% phase constrained least squares residual r=W*A*x*exp(i*phi)-b
function [r phi x JB JR] = pclsr(psi,te,b,opts)

% note echos in dim 1
[ne nx ny nz] = size(b);
b = reshape(b,ne,nx*ny*nz);
psi = reshape(psi,1,nx*ny*nz);

% change of variable

[tpsi dR2] = transform(psi,opts);

% complex fieldmap
W = exp(i*te*tpsi);

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

% smooth initial phase (phi)
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

% box constraint (0<=FF<=1)
if opts.nonnegFF
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

% if not needed, return early
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

% change variable
JR = bsxfun(@times,JR,dR2);

%% return arguments

x = reshape(x,2,nx,ny,nz);
JB = reshape(JB,ne,nx,ny,nz);
JR = reshape(JR,ne,nx,ny,nz);
phi = reshape(phi,1,nx,ny,nz);

%% change of variable for nonnegative R2*
function [psi dR2] = transform(psi,opts)

B0 = real(psi);
R2 = imag(psi);

if opts.nonnegR2
    dR2 = sign(R2) + (R2==0); % fix deriv=0 at R2=0
    R2 = abs(R2);
else
    dR2 = 1;
end

% transformed psi
psi = complex(B0,R2);

%% use phase unwrapping to remove 2pi boundaries
function psi = unswap(psi,te,data,opts)

% this unwraps in 3 passes: first for aliasing
% and second for fat-water swaps. then again
% to fix unwraps made on the wrong pass. it
% would be better if we could do a single pass
% and choose the best option.

% swaps at 2 frequences (Hz)
swap(1) = 1/min(diff(te)); % aliasing
swap(2) = -real(opts.psif)/2/pi; % fat-water
swap(3) = abs(diff(swap(1:2))); % both

% frequencies too close, skip the difference
if min(swap(3)./swap(1:2)) < 0.2; swap(3) = []; end

% only unwrap B0
B0 = real(squeeze(psi));

% unwrap is CPU only (mex)
B0 = gather(B0);
swap = gather(swap);
mask = gather(opts.mask);

% unwrap all swaps
for k = 1:numel(swap)

    B0 = B0/swap(k);
    if ndims(B0)==2
        B0 = unwrap2(B0,mask);
    else
        B0 = unwrap3(B0,mask);
    end
    B0 = B0*swap(k);
    
end

% remove gross aliasing (cosmetic)
alias_freq = 2*pi/min(diff(te)); % aliasing freq (rad/s)
center_freq = median(B0(opts.mask)); % center B0 (rad/s)
nwraps = fix(center_freq / alias_freq); % no. of wraps
B0(opts.mask) = B0(opts.mask)-gather(nwraps*alias_freq);

% back to complex
psi = reshape(B0,size(psi))+i*imag(psi);
