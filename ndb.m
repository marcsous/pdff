function [params sse] = ndb(te,data,Tesla,H2O)
%[params sse] = ndb(te,data,Tesla,H2O)
%
% Fat water separation with double bond estimation.
% WARNING - SLOW!! Only does single pixel fitting.
%
% Inputs:
%  te is a vector in seconds (ne)
%  data is a complex vector (ne)
%  Tesla is field strength (scalar)
%  H2O is the water ppm (optional)
%
% Outputs:
%  params.B0 is B0 (Hz)
%  params.R2 is R2* (1/s)
%  params.FF is FF (%)
%  params.PH is PH (rad)

display = true;

%% sizes

ne = numel(te);
te = reshape(te,ne,1);
data = reshape(data,ne,1);

%% initialize parameters

tmp = angle(dot(data(1:ne-1),data(2:ne)));
B0 = tmp/2/pi/mean(diff(te)); % B0 in Hz
R2 = 10; % R2* in (1/s)/2pi
NDB = 3; % no. double bonds

if nargin<4 || isempty(H2O)
    H2O = 4.7; % water freq in ppm
end

% make lsqnonlin happy
te = double(gather(te));
data = double(gather(data));
Tesla = double(gather(Tesla));
B0 = double(gather(B0));
R2 = double(gather(R2));
H2O = double(gather(H2O));

%% nonlinear fitting

x1 = [B0   R2 NDB];
LB = [-Inf  0   1];
UB = [+Inf 100  6];

opts = optimset('display','off');

% first estimate (assume B0 is on water peak)
[x1,sse1,~,~,~,~,J1] = lsqnonlin(@(x)myfun(x,te,data,Tesla,H2O),x1,LB,UB,opts);

% second estimate (assume B0 is on fat peak)
[~,psif] = fat_basis(te,Tesla,NDB,H2O);
x2(1) = x1(1)-real(psif);
x2(2) = max(x1(2)-imag(psif),0);
x2(3) = NDB;
[x2,sse2,~,~,~,~,J2] = lsqnonlin(@(x)myfun(x,te,data,Tesla,H2O),x2,LB,UB,opts);

% choose best solution
if sse1 < sse2
    x = x1;
    J = J1;
    sse = sse1;
else
    x = x2;
    J = J2;
    sse = sse2;
end

% get the linear(ish) terms (wf p)
[r wf] = myfun(x,te,data,Tesla,H2O);

% display
if display
    
    cplot(te,data,'o'); xlabel('te (s)');
    te = linspace(0,min(te)+max(te),10*ne);
    data = myfun([x(:);wf(:)],te,data,Tesla,H2O);
    hold on; cplot(te,data); hold off
    title(['FF=' num2str(100*real(wf(2)/sum(wf)),'%.1f') ' NDB=' num2str(x(3),'%.2f')]);
    axis tight; drawnow

end

% return variable
params.B0 = x(1);
params.R2 = 2*pi*x(2); % convert to 1/s
params.FF = min(max(100*real(wf(2)/sum(wf)),-10),110); 
params.PH = angle(sum(wf));
params.NDB = x(3);

% display
if display
    
    v = 2*ne-6; % no. degrees of freedom
    cov = pinv(full(J'*J))*sse/v; % covariance matrix
    ci95 = sqrt(max(diag(cov),0))*1.96; % confidence intervals
    x(2) = 2*pi*x(2); ci95(2) = 2*pi*ci95(2); R2 = 2*pi*R2; % 1/s
    
    disp([' initial B0 ' num2str(B0) ' R2* ' num2str(R2)])
    disp([' B0    ' num2str(x(1)) ' ± ' num2str(ci95(1))])
    disp([' R2*   ' num2str(x(2)) ' ± ' num2str(ci95(2))])
    disp([' FF    ' num2str(params.FF)])
    disp([' PH    ' num2str(params.PH)])
    disp([' NDB   ' num2str(x(3)) ' ± ' num2str(ci95(3))])
    disp([' H2O   ' num2str(H2O) ' (fixed)'])
    disp([' sse   ' num2str(sse,10)])
    disp(' ')
    
end

%% function to calculate residual
function [r wf] = myfun(x,te,data,Tesla,H2O)

% unknowns
B0 = x(1);  % field map
R2 = x(2);  % exp decay
NDB = x(3); % no. double bonds

% water/fat time evolution matrix
A = fat_basis(te,Tesla,NDB,H2O);

% fieldmap and R2*
W = diag(exp(2*pi*i*complex(B0,R2)*te));

% two paths: one for residual, one for display
if numel(x)==3

    % calculate water, fat and initial phase
    Mh = pinv(real(A'*W'*W*A));
    Ab = A'*W'*data; % tricky: W'*W*inv(W) = W'
    p = angle(sum(Ab.*(Mh*Ab)))/2;
    wf = Mh*real(Ab*exp(-i*p))*exp(i*p);

    % recalculate p to absorb sign of wf (cosmetic)
    if nargout>1
        p = angle(sum(wf));
        wf = real(wf*exp(-i*p))*exp(i*p);
    end
    
    % residual
    r = reshape(W*A*wf-data,size(te));
    r = [real(r);imag(r)];

else

    % display: function values from provided x
    wf = x(4:5);
    r = W*A*wf;

end
