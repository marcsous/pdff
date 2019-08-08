function out = ifft3(in,m,n,p,varargin)

if nargin<2; m = size(in,1); end
if nargin<3; n = size(in,2); end
if nargin<4; p = size(in,3); end
out = ifft2(in,m,n,varargin{:});
out = ifft(out,p,3,varargin{:});
