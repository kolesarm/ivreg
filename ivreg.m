function [beta se stats] = ivreg(y, T, Z, W, varargin)
%IVREG Instrumental Variables regression
%   BETA = IVREG(Y, T, Z, W) returns a vector BETA of different estimators of
%   the causal effect of the n-by-1 vector T on the n-by-1 vector out outcomes
%   Y, where the n-by-1 vector T is endogenous, and the n-by-L matrix W contains
%   exogenous regressors. Z is an n-by-K matrix of instruments. These matrices
%   may be sparse.
%
%   In particular, BETA = [OLS, TSLS, LIML, MBTSLS, JIVE, UJIVE, RTSLS]. LIML is
%   Limited information maximum likelihood, MBTSLS is the Kolesar, Chetty,
%   Friedman, Glaeser and Imbens (2011) modified bias-corrected two-stage least
%   squares, JIVE is the JIVE1 estimator in Angrist, Imbens, and Krueger (1999),
%   and UJIVE is the Kolesar (2012) version of the JIVE estimator. RTSLS is the
%   reverse two-stage least squares estimator.
%
%   [BETA, SE] = IVREG(Y, T, Z, W) returns an estimate standard errors of beta.
%   SE is a 4x7 matrix. The first row gives standard errors valid under
%   homoscedasticity and standard asymptotics (classic standard errors). The
%   second row computes robust standard errors for BETA that are valid under
%   heteroscedasticity. The third row computes many-instrument robust standard
%   errors for BETA that are valid under homoscedasticity and Bekker
%   asymptotics, allowing also for the presence of many covariates. The fourth
%   row computes estimates of the standard errors that are valid under the many
%   invalid instruments sequence of Kolesar, Chetty, Friedman, Glaeser and
%   Imbens (2011)
%
%   [BETA, SE, STATS] = IVREG(Y, T, Z, W) returns a cell array STATS of
%   additional statistics. The first column of STATS contains the values of the
%   statistics, and the second column contains their names (labels). The first
%   row returns the first-stage F statistic, labelled 'F', so that
%   STATS{1,1:2}={Fvalue,'F'}. The second row contains an estimate of the
%   reduced-form covariance matrix, labelled Omega. The third row contains an
%   estimate of XI, labelled 'Xi'. The fourth row returns the Sargan test of
%   overidentifying restrictions, STATS{5,1} is a 2-by-1 vector, with the first
%   element equal to the test statistic and the second element equal to the
%   p-value. The fifth row returns the Cragg-Donald test of overidentifying
%   restrictions, STATS{6,1} is a 2-by-1 vector, with the first element equal to
%   the test statistic and the second element equal to the p-value. Kolesar
%   (2012) modification. If there is only one instrument, STATS{5,1} and
%   STATS{6,1} return NaN.
%
%   [...] = IVREG(y, T,..., 'noConstant', BOOL,...) if false, adds a constant as
%   exogenous regressor unless W spans a constant vector already. Default is
%   false.
%
%   [...] = IVREG(y, T,..., 'printTable', BOOL,...) if true, prints the
%   estimation results in table form. Default is false.
%
%   See the documentation file ivreg.pdf for details on how BETA, SE, and STATS
%   are computed

% Time-stamp: <2013-03-12 17:25:34 (kolesarm)>
% Author:     <Michal Kolesár>
% Email:      <kolesarmi@googlemail dotcom>

%% 1.  First parse the inputs
p = inputParser;

p.addRequired('y',@isnumeric);
p.addRequired('T',@isnumeric);
p.addRequired('Z',@isnumeric);
p.addRequired('W',@isnumeric);

p.addParamValue('noConstant',true,@islogical);
p.addParamValue('printTable',false,@islogical);

p.parse(y, T, Z, W, varargin{:})

% Check that the matrices (T, Z, W) and left hand side (y) have compatible dimensions
[n,K] = size(Z);
L = size(W,2);
if ~isvector(y) | ~isvector(T)
    error('stats:ivreg:InvalidData', 'Y and T must be a vectors.');
elseif numel(y) ~= n | numel(T) ~= n | size(W,1) ~= n
    error('stats:ivreg:InvalidData', ...
          'The number of rows in Y, W and T must equal the number of rows in Z.');
elseif size(y,2) == n
    warning('stats:ivreg:ColumnVector', 'Y should be a column vector.');
    y = y';
elseif size(T,2) == n
    warning('stats:ivreg:ColumnVector', 'T should be a column vector.');
    T = T';
end

% If matrix W does not include a constant, add it
if p.Results.noConstant == false
    Wc = [W ones(n, 1)]; % matrix X with a constant added
    if issparse(W) && rank(full(Wc' * Wc)) > rank(full(W' * W)),
        W = Wc;
    elseif ~issparse(W) && rank(Wc' * Wc) > rank(W' * W),
        W = Wc;
    end
end

%% 2. Point estimates

MW = @(Z) Z - W*(W\Z); % annihilator

Yp = MW([y T]);
Zp = MW(Z); % Y_\perp, Z_\perp

YY  = full(Yp' * Yp); % [Y T]'*M_{W}*[Y T]: need this to be full to compute eigenvalues
YPY  = (Yp'*Zp)*(Zp\Yp); % [Y T]'*H_{W}*[Y T]:
YMY = YY - YPY; % ditto


%% 2.1 k-class: OLS, TSLS, LIML, MBTLS

k = [0 1 min(eig(YY / YMY)) (1-L/n)/(1-(K - 1)/n-L/n)];
beta = (YY(1,2)-k*YMY(1,2))./(YY(2,2)-k*YMY(2,2));

%% 2.2 JIVE, UJIVE

ZW     = [W Z];
MZW    = @(T) T - ZW*(ZW\T);

D      = sum(ZW/(ZW'*ZW) .*ZW,2);  % D=diag(P_ZW) as a vector
iIDZW  = ones(n,1)./(ones(n,1)-D); % (I-D)^{-1}
D      = sum(W/(W'*W) .*W,2);      % D=diag(P_W) as a vector
iIDW   = ones(n,1)./(ones(n,1)-D); % (I-D)^{-1}

hatTujive = T - iIDZW .* MZW(T);
hatPjive  = MW(hatTujive);
hatPujive = hatTujive - (T - iIDW .* MW(T));

betaLabels = {'OLS'; 'TSLS'; 'LIML'; 'MBTSLS'; 'JIVE'; 'UJIVE'; 'RTSLS'};
beta = [beta, (hatPjive'*y) ./ (hatPjive'*T), ...
        (hatPujive'*y) ./ (hatPujive'*T) YPY(1, 1)/YPY(1, 2)];


%% 5. Standard Errors
if nargout > 1 | p.Results.printTable == true,
    se = NaN(4,7);
    epsilon = @(beta) Yp(:,1) - Yp(:,2)*beta;

    %% 5.1 Homoscedastic
    sig = @(beta) epsilon(beta)'*epsilon(beta)/n;
    se(1,1:6) = sqrt([sig(beta(1))/(Yp(:,2)'*Yp(:,2)) ...
               [sig(beta(2)) sig(beta(3)) sig(beta(4))]/YPY(2,2) ...
               sig(beta(5))*(hatPjive'*hatPjive)/(hatPjive'*T)^2 ...
               sig(beta(6))*(hatPujive'*hatPujive)/(hatPujive'*T)^2]);

    %% 5.2 Heteroscedastic
    % ols
    se(2,1)= sqrt(sum((epsilon(beta(1)).*Yp(:,2)).^2)) / YY(2,2);

    % tsls, liml, mbtsls
    hatP  = Zp*(Zp\Yp(:,2));
    sekclass = @(beta) sqrt(sum((epsilon(beta).*hatP).^2)) / YPY(2,2);
    se(2,2:4)=[sekclass(beta(2)) sekclass(beta(3)) sekclass(beta(4))];

    % jive
    se(2,5) = sqrt(sum((epsilon(beta(5)).*hatPjive).^2)) / ...
              (hatPjive'*T);
    se(2,6) = sqrt(sum((epsilon(beta(6)).*hatPujive).^2)) / ...
              (hatPujive'*T);

    %% 5.3 Many instruments

    % Hessian of random-effects
    Sp = YMY/(n-K-L);
    S = YPY/n;
    lamre = max(eig(Sp\S))-K/n;
    a = [beta(3);1];
    b = [1;-beta(3)];
    Omre = (n-K-L)*Sp/(n-L) + n*(S-lamre*a*a'/((a'/Sp)*a))/(n-L);
    Qs = b'*S*b/(b'*Omre*b);
    c = lamre*Qs / ((1-L/n) * (K/n+lamre));

    se(3,3) = sqrt(-b'*Omre*b / (n*lamre) * (lamre+K/n) / ...
              (Qs*Omre(2,2)-S(2,2) + c/(1-c)*Qs/((a'/Omre)*a)));

    % mbtsls, using maximum URE likelihood plug-in estimator
    b = [1;-beta(4)]; % b_mbtsls
    Lam11 = max([0, b' * (S-K/n*Sp)* b]);

    if min(eig(Sp\S)) > K/n
        Lam22 = S(2, 2) - K/n*Sp(2, 2);
        Omure = Sp;
    else
        Lam22 = lamre/((a'/Omre)*a);
        Omure = Omre;
    end % if min(eig(Sp\S)) > K/n

    Gamma = @(beta) [1, 0; -beta, 1];
    Sig = Gamma(beta(4))'*Omure*Gamma(beta(4));
    h = (1-L/n) * (K-1)/n / (1-L/n-(K-1)/n);
    Vvalid = Sig(1,1)/Lam22 + h*(Sig(1,1)*Sig(2,2)+Sig(1,2)^2)/Lam22^2;
    Vinvalid = Vvalid + (Lam11*Omure(2,2) + Lam11*Lam22*n/K)/Lam22^2;

    se(3:4,4) = sqrt([Vvalid; Vinvalid]/n);

end % if nargout > 1,

%% 3. Other outputs

if nargout > 2
    stats=cell(5,2);
    stats{1, 1} = YPY(2, 2) / (K * Sp(2,2)); % first-stage F
    stats{2, 1} = Sp; % reduced-form covariance matrix
    stats{3, 1} = YPY/n - (K/n)* Sp; % Xi

    stats(:,2)={'F','Omega','Xi'};

    if size(Z, 2) > 1,
        overid(1) = N * (1 - ([1 -beta(3)] * YMY * [1;-beta(3)])...
                         / ([1 -beta(3)] * YY * [1;-beta(3)]));
        pvalue(1) = 1 - chi2cdf(overid(1),K - 1);
        overid(2) = (N - K - L) * log((N - K - L) * k(3) / (N - L))...
            - K * log((N - L) * overid(1) / (N * K));
        pvalue(2) = 1 - chi2cdf(overid(3),1);
    else
        overid = NaN(4, 1);
    end
    stats{4,1} = [];
    stats{4,2} = 'Sargan test statistic and p-value';
    stats{5,1} = [];
    stats{5,2} = 'Modified Cragg-Donald test statistic and p-value';

end % if nargout > 2


%% 6. Print results

if p.Results.printTable == true,
  for k = 1: length(beta)
      fprintf(' %6s:  % -.3f \n',betaLabels{k}, roundn(full(beta(k)),-3));
  end % for
  fprintf('\n%d observations, ',n);
  if K>1
      fprintf('%d instruments, ',K);
  else
      fprintf('1 instrument, ');
  end
  if L>1
      fprintf('%d covariates, ',L);
  else
      fprintf('1 covariate, ');
  end
  fprintf('first-stage F=%.1f \n', roundn(F,-1));
end %
