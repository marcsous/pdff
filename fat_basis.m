function [A psif] = fat_basis(te,Tesla,NDB,H2O,units)
% [A psif] = fat_basis(te,Tesla,NDB,H2O,units)
%
% Function that produces the fat-water matrix. Units can be adjusted
% to reflect the relative masses of water and fat. Default is proton.
% For backward compatibility, set NDB=-1 to use old Hamilton values.
%
% Inputs:
%   te (echo times in s)
%   Tesla (field strength)
%   NDB (number of double bonds)
%   H2O (water freq in ppm)
%   units ('proton' or 'mass')
%
% Ouputs:
%   A = matrix [water fat] of basis vectors
%   psif = best fit of fat to exp(2*pi*i*psif*te)
%
% Ref:
% Bydder M, Girard O, Hamilton G. Magn Reson Imaging. 2011;29:1041

%% argument checks

if max(te)>1
    error('''te'' should be in seconds.');
end
if ~exist('NDB','var') || isempty(NDB)
    NDB = 2.5;
end
if ~exist('H2O','var') || isempty(H2O)
    H2O = 4.7;
end
if ~exist('units','var') || isempty(units)
    units = 'proton';
end

%% triglyderide properties

if NDB == -1
    % backward compatibility
    d = [1.300 2.100 0.900 5.300 4.200 2.750];
    a = [0.700 0.120 0.088 0.047 0.039 0.006];
    awater = 1;
else
    % fat chemical shifts in ppm
    d = [5.29 5.19 4.2 2.75 2.2 2.02 1.6 1.3 0.9];
    
    % Mark's heuristic formulas
    CL = 16.8+0.25*NDB;
    NDDB = 0.093*NDB^2;
    
    % Gavin's formulas (no. protons per molecule)
    awater = 2;
    a(1) = NDB*2;
    a(2) = 1;
    a(3) = 4;
    a(4) = NDDB*2;
    a(5) = 6;
    a(6) = (NDB-NDDB)*4;
    a(7) = 6;
    a(8) = (CL-4)*6-NDB*8+NDDB*2;
    a(9) = 9;
    
    % the above counts no. molecules so that 1 unit of water = 2
    % protons and 1 unit of fat = sum(a) = (2+6*CL-2*NDB) protons.
    if isequal(units,'mass')
        % scale in terms of molecular mass so that 18 units of
        % water = 2 protons and (134+42*CL-2*NDB) units of fat =
        % (2+6*CL-2*NDB) protons. I.e. w and f are in mass units.
        awater = awater/18;
        a = a/(134+42*CL-2*NDB);
    else
        % scale by no. protons (important for PDFF) so 1 unit
        % of water = 1 proton and 1 unit of fat = 1 proton.
        awater = awater/2;
        a = a/sum(a);
    end
end

%% fat water matrix

% put te in the right format
ne = numel(te);
te = reshape(te,ne,1);
te = gather(te);

% time evolution matrix
fat = te*0;
water = te*0+awater;
larmor = 42.57747892*Tesla; % larmor freq (MHz)
for j = 1:numel(d)
    freq = larmor*(d(j)-H2O); % Hz relative to water
    fat = fat + a(j)*exp(2*pi*i*freq*te);
end
A = [water fat];

%% best fit of fat to complex exponential (gauss newton)

if nargout>1
    psif = complex(-150*Tesla,10);
    for j = 1:10
        r = myfun(psif,te,fat);
        h = 1e-4;
        J = (myfun(psif+h,te,fat)-r)/h;
        psif = psif - pinv(J)*r;
    end
end

% exponential fitting function
function r = myfun(psif,te,data)
f = exp(2*pi*i*psif*te);
v = (f'*data)/(f'*f); % varpro
r = v*f-data; % residual
