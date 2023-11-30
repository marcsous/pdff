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

% phase roll along readout (unit: 2pi phase cycles <=> kspace points)
roll = i * linspace(-pi,pi,size(data,dim));
if dim==1; roll = reshape(roll,[],1,1); end
if dim==2; roll = reshape(roll,1,[],1); end
if dim==3; roll = reshape(roll,1,1,[]); end

% phase matrix
s = size(data); s(dim) = 1;
P = repmat(cast(roll,'like',data),s);
if ndims(data)==3; P(:,:,  2:2:end) = -P(:,:,  2:2:end); end
if ndims(data)==4; P(:,:,:,2:2:end) = -P(:,:,:,2:2:end); end

% minimize cost function
opts = optimset('Display','off','GradObj','on');
[phi,~,~,~,~,H] = fminunc(@(phi)myfun(phi,data,P),0,opts);

% 95% confidence interval
ci95 = sqrt(diag(inv(abs(H)))) * 1.96;

fprintf('Gradient delay = %f ± %f dwell times\n',phi,ci95);

% get corrected data
[~,~,data] = myfun(phi,data,P);

%% cost function: nuclear norm of A
function [nrm grd A] = myfun(phi,data,P)

% phase correct data
A = exp(phi*P) .* data;

% matrix of echo variation in all pixels (doi.org/10.1016/j.mri.2006.03.006)
A = reshape(A,[],size(data,ndims(data)));

% nuclear norm and derivative w.r.t. phi
if nargout==1

    W = svd(A'*A);
    W = sqrt(W);
    nrm = gather(sum(W,'double'));
    
else
    
    [V W] = svd(A'*A);
    W = sqrt(diag(W));
    dA = A.*reshape(P,size(A));
    dW = real(diag(V'*(A'*dA)*V))./W;
    nrm = gather(sum( W,'double'));
    grd = gather(sum(dW,'double'));

end

% return phase corrected data
A = reshape(A,size(data));
