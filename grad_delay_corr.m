function [data phi] = grad_delay_corr(data,dim)
%
% a simple bipolar gradient delay correction
%
% uses nuclear norm to quantify the "amount" of signal variation
% along the echo dimension and compresses it w.r.t. a +/- phase
% roll (gradient delay) along the readout direction
%
% -data: 2D or 3D complex multi-echo images (echos in last dim)
% -dim: the readout dimension (default 2)
%
% Refs.
% doi.org/10.1016/j.mri.2006.03.006
% doi.org/10.1016/j.mri.2022.08.017

% demo dataset
if nargin==0
    load liver_12echo_bipolar.mat
end

% arg check
if ndims(data)<3 || ndims(data)>4 || isreal(data) || ~isfloat(data)
    error('data size/type not supported');
end

% readout dimension
if nargin<2
    dim = 2;
elseif ~ismember(dim,[1 2 3])
    error('dim is not supported');
end

% minimize cost function
opts = optimset('Display','off','GradObj','on');
[phi,~,~,~,~,H] = fminunc(@(phi)myfun(phi,data,dim),0,opts);

% 95% confidence interval
ci95 = sqrt(diag(inv(abs(H)))) * 1.96;

for k = 1:numel(phi)
    fprintf('Gradient delay = %f ± %f dwell times\n',phi(k),ci95(k));
end

% get corrected data
[~,~,data] = myfun(phi,data,dim);

%% cost function: nuclear norm of A
function [nrm grd data] = myfun(phi,data,dim)

% phase roll along readout (unit: 2pi phase cycle <=> 1 kspace point <=> 1 dwell time)
roll = i * linspace(-pi,pi,size(data,dim));
if dim==1; roll = reshape(roll,[],1,1); end
if dim==2; roll = reshape(roll,1,[],1); end
if dim==3; roll = reshape(roll,1,1,[]); end

% phase matrix
s = size(data); s(dim) = 1;
P = repmat(cast(roll,'like',data),s);

% alternate on odd/even echos
if ndims(data)==3; P(:,:,  2:2:end) = -P(:,:,  2:2:end); end
if ndims(data)==4; P(:,:,:,2:2:end) = -P(:,:,:,2:2:end); end

% phase correct data
data = exp(phi*P) .* data;

% matrix of echo variation in all pixels
A = reshape(data,[],s(end));

% nuclear norm and derivative w.r.t. phi
if nargout==1
    
    W = svd(A);
    dW = [];
    
else
    
    [U W V] = svd(A,'econ');
	W = diag(W);
    dA = A.*reshape(P,size(A));
    dW = real(diag((U'*dA)*V));

end

% plain doubles for fminunc
nrm = gather(sum( W,'double'));
grd = gather(sum(dW,'double'));
