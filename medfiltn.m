function B = medfiltn(A,P,mask)
%
% Median filter of N-D array using kernel P.
% Performs circular wrapping at boundaries.
%
% A is an N-D array
% P is a kernel (default [3 3])
% mask (binary) excludes points

% argument checks
if nargin<2 || isempty(P)
    P = [3 3];
elseif ~isvector(P) || any(P<1) || nnz(mod(P,1))
    error('something wrong with P');
end
if ~isreal(A)
    warning('median not well defined for complex values');
end
if numel(P)>ndims(A)
    error('P has more dims than A');
else
    P = reshape(P,numel(P),1);
end
if nargin>2
    if ~isequal(size(A),size(mask)) || nnz(mod(mask,1))
        error('something wrong with mask');
    end
    origA = A;
    A(~mask) = NaN; % let's us use omitnan flag
end

% generate shift indicies (S)
switch numel(P)
    case 1; [S(1,:)] = ind2sub(P,1:prod(P));
    case 2; [S(1,:) S(2,:)] = ind2sub(P,1:prod(P));
    case 3; [S(1,:) S(2,:) S(3,:)] = ind2sub(P,1:prod(P));
    case 4; [S(1,:) S(2,:) S(3,:) S(4,:)] = ind2sub(P,1:prod(P));
    case 5; [S(1,:) S(2,:) S(3,:) S(4,:) S(5,:)] = ind2sub(P,1:prod(P));
    otherwise; error('high dimensions not implemented - fix me');
end

% make data matrix
B = zeros(numel(A),prod(P),'like',A);

for k = 1:prod(P)
    tmp = circshift(A,S(:,k)-fix(P/2)-1);
    B(:,k) = reshape(tmp,numel(A),1);
end

% median along shift dim - for complex maybe use medoid?
B = median(B,2,'omitnan');

% use orig values when all masked
if nargin>2
    k = isnan(B);
    B(k) = origA(k);
end

% return it as we found it
B = reshape(B,size(A));
