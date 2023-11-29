function data = grad_delay_corr(data,xdim)
%
% a simple gradient delay correction
%
% uses nuclear norm to quantify "amount" of signal variation
% along the echo dimension and compress it w.r.t. a phase roll
% (gradient delay) along the x-direction.
%
% -data is a 2D or 3D complex multi-echo dataset (echos in last dim)
% -xdim is the readout dimension (default 2)

% demo dataset
if nargin==0
    load ~/octave/IDEAL.mat
end

% arg check
if ndims(data)<3 || ndims(data)>4 || isreal(data) || ~isfloat(data)
    error('data size/type not supported');
end

% readout dimension
if nargin<2
    xdim = 2;
elseif ~ismember(xdim,[1 2 3])
    error('xdim is not supported');
end

% gradient delay correction (assume same in all slice locations)
fopts = optimset('Display','off','GradObj','on');
phi = fminunc(@(phi)myfun(phi,data,xdim),0,fopts);

% get corrected data
[~,~,data] = myfun(phi,data,xdim);

%% cost function: nuclear norm
function [nrm grd data] = myfun(phi,data,xdim)

sz = size(data);

% phase roll along readout direction
roll = i * linspace(-1,1,sz(xdim));

roll = cast(roll,'like',data);
if xdim==1; roll = reshape(roll,[],1,1); end
if xdim==2; roll = reshape(roll,1,[],1); end
if xdim==3; roll = reshape(roll,1,1,[]); end

% phase matrix
sz(xdim) = 1;
P = repmat(roll,sz);
if ndims(data)==3; P(:,:,2:2:end) = -P(:,:,2:2:end); end
if ndims(data)==4; P(:,:,:,2:2:end) = -P(:,:,:,2:2:end); end

% phase corrected data matrix
A = reshape(exp(phi*P).*data,[],sz(4));

if nargout==1
    
    % nuclear norm (double for fminunc)
    W = svd(A'*A);
    W = sqrt(diag(W));
    nrm = gather(sum( W,'double'));
    
else
    
    % derivative required
    [V W] = svd(A'*A);
    W = sqrt(diag(W));
    dA = A.*reshape(P,size(A));
    dW = real(diag(V'*(A'*dA)*V))./W;
    nrm = gather(sum( W,'double'));
    grd = gather(sum(dW,'double'));
    
    if nargout>2
        % phase corrected data
        data = reshape(A,size(data));
    end

end
