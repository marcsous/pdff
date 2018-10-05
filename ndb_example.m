function params = ndb_example()

%% phantom data from paper
load PHANTOM_NDB_PAPER.mat

H2O = 4.8; % water at room temp is 4.8~4.9 ppm
data = conj(data); % GE data is complex conjugate

[nx ny ne] = size(data);
params = zeros(nx,ny,5);

% loop over every pixel
for x = 1:nx
    for y = 1:ny
      
        ims(params,[],{'B0','R2*','NDB','FF','PHI'});
        params(x,y,:) = fit_ndb(te,data(x,y,:),Tesla,[],H2O);

    end
end

keyboard

%% fitting function [B0 R2 NDB]
function [x sse] = fit_ndb(te,data,Tesla,B0,H2O)

% Fat water separation with double-bond fitting.
% WARNING - SLOW!! Only does a single pixel.
%
% te is a vector in seconds (ne)
% data is a complex vector (ne)
% Tesla is field strength (scalar)
% B0 is an initial estimate for B0 (optional)
% H2O is the water ppm (optional)

display = true;

%% sizes

ne = numel(te);
te = reshape(te,ne,1);
data = reshape(data,ne,1);

%% initialize parameters

NDB = 3; % no. double bonds
R2 = 20; % default R2* in 1/s

if nargin<4 || isempty(B0)
    dt = te(2)-te(1); % assume constant spacing
    temp = dot(data(1:ne-1),data(2:ne));
    B0 = angle(temp)/2/pi/dt; % B0 in Hz
end
if nargin<5 || isempty(H2O)
    H2O = 4.7; % water freq in ppm
end

% make lsqnonlin happy
te = double(te);
data = double(data);
Tesla = double(Tesla);
B0 = double(B0);
R2 = double(R2);
H2O = double(H2O);

%% nonlinear fitting

x1 = [B0   R2  NDB];
LB = [-Inf  0   1 ];
UB = [+Inf 1000 6 ];

opts = optimset('display','off');

% first estimate (assume B0 is on water peak)
[x1,sse1,~,~,~,~,J1] = lsqnonlin(@(x)myfun(x,te,data,Tesla,H2O),x1,LB,UB,opts);

% second estimate (assume B0 is on fat peak)
[~,psif] = fat_basis(te,Tesla,NDB,H2O);
x2(1) = x1(1)-real(psif);
x2(2) = max(x1(2)-2*pi*imag(psif),0);
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

% get the linear terms (wf)
[~,wf] = myfun(x,te,data,Tesla,H2O);

% display
if display
    
    cplot(te,data,'o');
    hold on
    te = linspace(0,max(te),10*ne);
    data = myfun([reshape(x,[],1);wf],te,data,Tesla,H2O);
    cplot(te,data);
    hold off
    axis tight
    drawnow

end

% recalculate p to absorb sign of wf
p = angle(sum(wf));
wf = real(wf*exp(-i*p));

% convert wf to FF 
x(4) = wf(2)/sum(wf);

% limit FF values (for display)
x(4) = max(min(x(4),1.1),-0.1);

% initial phase
x(5) = p;

% display
if display
    
    v = 2*ne-numel(x); % degrees of freedom
    cov = pinv(full(J'*J))*sse/v; % covariance matrix
    ci95 = sqrt(max(diag(cov),0))*1.96; % 95% confidence intervals
    
    disp([' initial B0 ' num2str(B0) ' R2* ' num2str(R2)])
    disp([' B0    ' num2str(x(1)) ' ± ' num2str(ci95(1))])
    disp([' R2*   ' num2str(x(2)) ' ± ' num2str(ci95(2))])
    disp([' NDB   ' num2str(x(3)) ' ± ' num2str(ci95(3))])
    disp([' FF    ' num2str(x(4))])
    disp([' PHI   ' num2str(x(5))])
    disp([' H2O   ' num2str(H2O) ' (fixed)'])
    disp([' sse   ' num2str(sse,10)])
    disp(' ')
    
end

%% function to calculate residual
function [r wf] = myfun(x,te,data,Tesla,H2O)

% unknowns
B0 = x(1);     % field map Hz
R2 = x(2);     % exp decay 1/s
NDB = x(3);    % no. double bonds

% water/fat time evolution matrix
A = fat_basis(te,Tesla,NDB,H2O);

% fieldmap and R2*
W = diag(exp((2*pi*i*B0-R2)*te));

% two paths: one for residual, one for display
if numel(x)==3

    % calculate water, fat and initial phase
    Mh = inv(real(A'*W'*W*A));
    Ab = A'*W'*data; % tricky: W'*W*inv(W) = W'
    p = angle(sum(Ab.*(Mh*Ab)))/2;
    wf = Mh*real(Ab*exp(-i*p))*exp(i*p);

    % residual
    r = reshape(W*A*wf-data,size(te));
    r = [real(r);imag(r)];

else

    % generate function values from provided x
    wf = x(4:5);
    r = W*A*wf;

end
