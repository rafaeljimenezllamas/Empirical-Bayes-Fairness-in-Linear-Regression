function [Final_Solution] = Code_3_ZP(X,y,z,eps,NumT,NumIter,NumFev,NumPoints,a_ini,b_ini,beta0_ini,g_ini)

[N,p] = size(X);
p = p-1;

% Log Marginal Likelihood

fun = @(x) aux_fun(x,X,y,z,eps);


global Xaux yaux zaux epsaux

Xaux=X;
yaux=y;
zaux=z;
epsaux=eps;

% Initial values

% Initial values

x_ini = zeros(p+1,3);
x_ini(1,1) = a_ini;
x_ini(1,2) = b_ini;
x_ini(2,1) = g_ini;
x_ini(:,3) = beta0_ini;

% Lower bounds

LB = zeros(1,3*(p+1));
LB(1) = 1e-3;               % Lower bound for a
LB(2) = 1e-5;               % Lower bound for g
LB(p+2) = 1e-3;             % Lower bound for b
LB(2*p+3:end) = -10^4;    % Lower bound for beta0_i

% Upper bounds

UB = zeros(1,3*(p+1));
UB(1) = 1e5;                % Upper bound for a
UB(2) = 1e5;                % Upper bound for g
UB(p+2) = 1e5;              % Upper bound for b
UB(2*p+3:end) = 10^4;     % Upper bound for beta0_i

            
% Optimization problem

problem = createOptimProblem('fmincon','objective', ...
          @(x) fun(x), 'x0', x_ini, 'lb', LB, 'ub', UB,'nonlcon',@nonlcon);
problem.options.MaxIterations=NumIter;
problem.options.MaxFunctionEvaluations=NumFev;
gs = GlobalSearch('Display','iter','StartPointsToRun','bounds-ineqs','MaxTime',NumT,'NumTrialPoints',NumPoints);
[Sol, f] = run(gs, problem);


% Solution

a_opt = Sol(1,1);               % Value of a at solution
b_opt = Sol(1,2);               % Value of b at solution
g_opt = Sol(2,1);               % Value of g at solution
beta0_opt = Sol(:,3);           % Value of beta0 at solution
invSigma_opt = 1/g_opt*(X'*X);     % Value of Sigma at solution

Sigma1 = inv(X'*X+invSigma_opt);                  % We calculate Sigma1          
beta1 = Sigma1*(X'*y + invSigma_opt*beta0_opt);   % We calculate beta1 

a1 = a_opt + N;
b1 = b_opt + y'*y + beta0_opt'*invSigma_opt*beta0_opt-beta1'*inv(Sigma1)*beta1;   % We do the same with b1.


% We change sign to find the actual value of the marginal likelihood we
% want to maximize
f=-f;

eps_opt = abs(z'*beta1);


if eps_opt>eps
    f=-10^20;       % We penalize unfulfilling the fairness constraint
end


Final_Solution = {Sol,a1,b1,beta1,Sigma1,f};


%% Auxiliary function

function [c,ceq] = nonlcon(x)
    a = x(1,1);
    b = x(1,2);
    g = x(2,1);
    beta0 = x(:,3);
    invSigma = 1/g*Xaux'*Xaux;

    Sigma1_aux = inv(Xaux'*Xaux+invSigma);
   
    beta1_aux = Sigma1_aux*(Xaux'*yaux+invSigma*beta0);

    c=abs(zaux'*beta1_aux)-epsaux;
    ceq=[];
end


function sol = aux_fun(x,X,y,z,eps)
    a = x(1,1);
    b = x(1,2);
    g = x(2,1);
    beta0 = x(:,3);
    invSigma = 1/g*Xaux'*Xaux;

    [N_aux,p_aux] = size(X);

    Sigma1_aux = inv(X'*X+invSigma);
   
    beta1_aux = Sigma1_aux*(X'*y+invSigma*beta0);

    b1_aux = b + y'*y + beta0'*invSigma*beta0 - beta1_aux'*inv(Sigma1_aux)*beta1_aux;

    term1 = +1/2*log(det(eye(p_aux)+inv(invSigma)*(X'*X)));
    term2 = -a/2*log(b/2)+(a+N_aux)/2*log(b1_aux/2);
    term3 = -sum(log(a/2+(0:(floor(N_aux/2)-1))));

    sol=term1+term2+term3;     % Sum of terms of (log) marginal likelihood

end

end