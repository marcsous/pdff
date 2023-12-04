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
elseif ~isscalar(dim) || ~ismember(dim,[1 2 3])
    error('dim is not supported');
end

% minimize cost function (DerivativeCheck requires doubles to pass)
opts = optimset('Display','off','GradObj','on','DerivativeCheck','off');
[phi,~,~,~,~,H] = fminunc(@(phi)myfun(phi,data,dim),0,opts);

% 95% confidence interval
ci95 = sqrt(diag(inv(abs(H)))) * 1.96;

fprintf('Gradient delay = %f ± %f dwell times\n',phi,ci95);

% get corrected data
[~,~,data] = myfun(phi,data,dim);

%% cost function: nuclear norm of A
function [nrm grd data] = myfun(phi,data,dim)

% key size parameters
siz = size(data);
nx = siz(dim);
ne = siz(end);

% matrix of echo variation in all pixels
A = reshape(data,[],ne);

% phase roll along readout (unit: 2pi phase cycle <=> 1 kspace point <=> 1 dwell time)
roll = cast(i * linspace(-pi,pi,nx),'like',data);
if dim==1; roll = reshape(roll,[],1,1); end
if dim==2; roll = reshape(roll,1,[],1); end
if dim==3; roll = reshape(roll,1,1,[]); end

% phase matrix
siz(dim) = 1;
P = repmat(roll,siz);
P = reshape(P,[],ne);

% alternate odd/even echos
P(:,2:2:ne) = -P(:,2:2:ne);

% phase correct data
A = exp(phi*P) .* A;

% nuclear norm and derivative w.r.t. phi
if nargout==1
    
    W = svd(A);
    dW = [];
    
else
    
    [U W V] = svd(A,'econ');
	W = diag(W);
    dA = A.*P;
    dW = real(diag((U'*dA)*V));

    % return phase corrected data
    data = reshape(A,size(data));
    
end

% plain doubles for fminunc
nrm = gather(sum( W,'double'));
grd = gather(sum(dW,'double'));
