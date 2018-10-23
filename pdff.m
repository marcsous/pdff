function [params sse] = pdff(te,data,Tesla,varargin)
%[params sse] = pdff(te,data,Tesla,varargin)
%
% Proton density fat fraction estimation using phase
% constrained least squares. Returns [B0 R2* FF PHI].
%
% Works best with phase unwrapping (e.g. unwrap2.m &
% unwrap3.m from https://github.com/marcsous/unwrap).
%
% Inputs:
%  te is echo time in seconds (vector)
%  data (1D, 2D or 3D with echos in last dimension)
%  Tesla field strength (scalar)
%  varargin in option/value pairs (e.g. 'ndb',3.0)
%
% Alternate input:
%  imDataParams is a struct from fat-water toolbox
%
% Outputs:
%  params.B0 is B0 (Hz)
%  params.R2 is R2* (1/s)
%  params.FF is FF (%)
%  params.PH is PH (rad)
%  sse is sum of squares error

% demo code
if nargin==0
    load PHANTOM_NDB_PAPER.mat
    varargin = {'h2o',4.8,'ndb',3}; % phantom options
end

%% default options

opts.ndb = 2.5; % no. double bonds
opts.h2o = 4.7; % water frequency ppm
opts.maxit = 10; % max. no. iterations
opts.noise = []; % noise std (if available)
opts.filter = [5 5 3]; % smoothing filter size

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

%% special handling to accept ISMRM F/W toolbox structure

if isa(te,'struct')
    Tesla = te.FieldStrength;
    [nx ny nz nc ne] = size(te.images);
    if nc>1; error('multiple coils not supported'); end
    data = reshape(te.images,nx,ny,nz,ne);
    if te.PrecessionIsClockwise<0; data = conj(data); end
    te = te.TE; % seconds
end

%% argument checks - be flexible: 1D, 2D or 3D data

if numel(data)==numel(te)
    data = reshape(data,1,1,1,[]);
elseif ndims(data)==2 && size(data,2)==numel(te)
    data = permute(data,[1 3 4 2]);
elseif ndims(data)==3 && size(data,3)==numel(te)
    data = permute(data,[1 2 4 3]);
elseif ndims(data)==4 && size(data,4)==numel(te)
else
    error('data [%s\b] not compatible with te [%s\b].',...
    sprintf('%i ',size(data)),sprintf('%i ',size(te)));
end
if numel(te)<3
    error('too few echos (%i) for B0, FF, R2* estimation.',numel(te));
end
if max(te)>1
    error('''te'' should be in seconds.');
end
if ~issorted(te)
    warning('''te'' should be sorted. Sorting.');
    [te k] = sort(te); data = data(:,:,:,k);
end
if isreal(data) || ~isfloat(data)
    warning('data should be complex float. Converting to single.');
    data = single(data);
end

%% set up some constants

% estimate noise std
if numel(data)==numel(te)
    opts.noise = 0;
elseif isempty(opts.noise)
    % need better way to estimate std
    S = svd(reshape(data,[],numel(te)));
    opts.noise = S(end)/sqrt(nnz(data)/numel(te));
end

% time evolution matrix
[opts.A opts.psif] = fat_basis(te,Tesla,opts.ndb,opts.h2o);

% display
disp([' Data size: [' sprintf('%i ',size(data)) sprintf('\b]')])
disp([' TE (ms): ' sprintf('%.2f ',1000*te(:))])
disp([' Tesla (T): ' sprintf('%.2f',Tesla)])
disp(opts);

%% crop rows/cols/slices to help cpu

mask = any(abs(data)>opts.noise,4);
x = any(any(mask,2),3);
y = any(any(mask,1),3);
z = any(any(mask,1),2);
if nnz(mask)==0; error('All pixels are zero.'); end

data = data(x,y,z,:);
[nx ny nz ne] = size(data);
fprintf(' Cropping images to [%i %i %i]\n',[nx ny nz]);

%% center kspace (otherwise smoothing is ineffective)

data = fft3(data);
tmp = sum(abs(data),4);
[~,k] = max(tmp(:));
[dx dy dz] = ind2sub([nx ny nz],k);
data = circshift(data,1-[dx dy dz]);
data = ifft3(data);
fprintf(' Shifting kspace center [%i %i %i]\n',dx,dy,dz);

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

%% echos in 1st dimension (faster dot products)

data = permute(data,[4 1 2 3]);
te = reshape(real(cast(te,'like',data)),ne,1);

%% nonlinear estimation (local optmization)

% initialize with dominant frequency (Hz)
tmp = dot(data(1:end-1,:,:,:),data(2:end,:,:,:),1);
psi = angle(tmp)/2/pi/mean(diff(te))+10i;

% find the 2 local minima
fprintf(' 1st local min\t');
[psi1 r1] = nlsfit(psi,te,data,opts);

fprintf(' 2nd local min\t');
[psi2 r2] = nlsfit(psi1-opts.psif,te,data,opts);

% choose the lowest ||r||
ok = dot(r1,r1) < dot(r2,r2);
psi(ok) = psi1(ok);
psi(~ok) = psi2(~ok);

% phase unwrapping to remove fat-water swaps
try
    tmp = gather(real(squeeze(psi)))*2*pi/real(opts.psif);
    if nx>1 && ny>1
        if nz==1
            tmp = unwrap2(tmp);
        else
            tmp = unwrap3(tmp);
        end
    end
    psi = reshape(tmp*real(opts.psif)/2/pi,size(psi))+i*imag(psi);
    fprintf(' Phase unwrapping succeeded.\n');
catch ME
    fprintf(' Phase unwrapping failed (%s).\n',ME.message);
end

%% ideal processing to remove remaining swaps

for iter = 1:opts.maxit
    
    fprintf(' IDEAL step %i\t',iter);
    
    % median filtering
    tmp = gather(squeeze(real(psi)));
    if nz==1
        filter = opts.filter(1:2);
        tmp = medfilt2(tmp,filter,'symmetric');
    else
        filter = opts.filter(1:3);
        tmp = medfilt3(tmp,filter,'symmetric');
    end
    psi = reshape(tmp,size(psi))+i*imag(psi);
    
    % mu constrains psi to initial value
    opts.mu = opts.noise * (iter/opts.maxit);
    [psi r p wf] = nlsfit(psi,te,data,opts);
    
    % single pixel fitting (no smoothing)
    if numel(data)==numel(te); break; end
end

%% return parameters in correct format

tmp = zeros(size(mask),'like',gather(te));

tmp(x,y,z) = gather(real(psi));
params.B0 = tmp; % B0 (Hz)

tmp(x,y,z) = gather(imag(psi)*2*pi);
params.R2 = tmp; % R2* (1/s)

tmp(x,y,z) = gather(100*wf(2,:,:,:)./sum(wf)); 
params.FF = tmp; % FF (%)

tmp(x,y,z) = gather(p);
params.PH = tmp; % initial phase (radians)

tmp(x,y,z) = gather(real(dot(r,r)));
sse = tmp;


%% nonlinear least squares fitting
function [psi r p wf] = nlsfit(psi,te,data,opts)

% constants
h = sqrt(eps('single')); % derivatives
tiny = realmin('single'); % division by 0
psi0 = psi; % use initial value to constrain
if ~isfield(opts,'mu'); opts.mu = 0; end
tic;

% gauss newton
for iter = 1:opts.maxit

    % residual
    r = pclsr(psi,te,data,opts);
    
    % Jacobian (separate real and imag parts)
    hr = max(abs(real(psi)),1)*h;
    hi = max(abs(imag(psi)),1)*h*i;
    Jr = bsxfun(@times,pclsr(psi+hr,te,data,opts)-r,1./hr);
    Ji = bsxfun(@times,pclsr(psi+hi,te,data,opts)-r,1./hi);
    
    % Jacobian J = [Jr i*Ji] (phase constrained least squares)
    % has 2x2 normal equations: Re(J'*J) = [J1 J2;J2 J3]
    J1 = real(dot(Jr,Jr))+opts.mu^2;
    J2 =-imag(dot(Jr,Ji));
    J3 = real(dot(Ji,Ji))+opts.mu^2;

    % solution [Re(Δpsi) Im(Δpsi)] = -inv(Re(J'*J))*Re(J'*r)
    % where inv(Re(J'*J)) = [J3 -J2;-J2 J1]/dtm (determinant)
    Jrr = real(dot(Jr,r))+opts.mu^2*real(psi-psi0);
    Jir = imag(dot(Ji,r))+opts.mu^2*imag(psi-psi0);
    dtm = max(J1.*J3-J2.^2,tiny); % dtm>0 for pos def matrix
    psi = psi-complex(J3.*Jrr-J2.*Jir,J1.*Jir-J2.*Jrr)./dtm;
    
    % prevent values that give Inf/NaN (mu should handle this...)
    psi = real(psi) + i*min(max(imag(psi),0),100);

end

% final call to get wf and p
[r p wf] = pclsr(psi,te,data,opts);

% display
if numel(data)==numel(te)
    cplot(te,data,'o');hold on;cplot(te,data-r);hold off;xlabel('te (s)');pause(0.1);
    title(['||r||=' num2str(norm(r)) ' psi=' num2str(psi,'%.1f') ' FF=' num2str(wf(2)/sum(wf),'%.3f')]);
    fprintf('B0=%.2f\tR2=%.3f\tFF=%.3f\tPH=%.4f\tsse=%.5e\n',real(psi),2*pi*imag(psi),100*wf(2)/sum(wf),p,norm(r)^2);
else
    fprintf('\t||r||=%.4e\tmu=%.4e\t',norm(r(:)),opts.mu); toc
    mid = round(size(data,4)/2); % middle slice
    tmp = zeros(size(data,2),size(data,3),4,'like',te);
    tmp(:,:,1) = p(1,:,:,mid); % initial phase
    tmp(:,:,2) = min(max(real(psi(1,:,:,mid)),-2.1*abs(opts.psif)),2.1*abs(opts.psif)); % B0
    tmp(:,:,4) = min(max(imag(psi(1,:,:,mid))*2*pi,-1),501); % R2*
    tmp(:,:,3) = min(max(100*wf(2,:,:,mid)./sum(wf(:,:,:,mid)),-5),105); % FF
    ims(tmp,1,{'PH (rad)','B0 (Hz)','FF (%)','R2* (1/s)'}); for k = 1:4; subplot(2,2,k); colorbar; end; drawnow
end

%% phase constrained least squares r = b-W*A*x*exp(i*p)
function [r p wf] = pclsr(psi,te,b,opts)

% note echos in dim 1
[ne nx ny nz] = size(b);

% to prevent div by 0
tiny = realmin('single');

% complex weighting (R2* and fieldmap)
W = exp(2*pi*i*te*reshape(psi,1,nx*ny*nz));

% (tricky) Ab = A'*W'*W*inv(W)*b = A'*W'*b
Ab = opts.A'*(conj(W).*reshape(b,ne,nx*ny*nz));

% Re(A'*W'*W*A) is a 2x2 matrix given by [A1 A2;A2 A3]
% with inverse [A3 -A2;-A2 A1]/dtm (determinant)
WW = real(W).^2+imag(W).^2;
A1 = real(conj(opts.A(:,1)).*opts.A(:,1))' * WW;
A2 = real(conj(opts.A(:,1)).*opts.A(:,2))' * WW;
A3 = real(conj(opts.A(:,2)).*opts.A(:,2))' * WW;
dtm = max(A1.*A3-A2.^2,tiny); % dtm>0 for pos def matrix

% solve Re(A'*W'*W*A)*x = Re(A'*W'*b*exp(-i*p))
p = angle(A3.*Ab(1,:).^2+A1.*Ab(2,:).^2-2*A2.*prod(Ab))/2;
Ab = real(bsxfun(@times,Ab,exp(-i*p)));
wf = [A3.*Ab(1,:)-A2.*Ab(2,:);A1.*Ab(2,:)-A2.*Ab(1,:)];
wf = bsxfun(@times,wf,exp(i*p)./dtm);
r = reshape(b,ne,nx*ny*nz)-W.*(opts.A*wf);

% recalculate p to absorb sign of x (cosmetic)
if nargout>1
    p = angle(sum(wf));
    wf = real(bsxfun(@times,wf,exp(-i*p)));
end

% residual, phase and water-fat estimates
r = reshape(r,ne,nx,ny,nz);
wf = reshape(wf,2,nx,ny,nz);
p = reshape(p,1,nx,ny,nz);
