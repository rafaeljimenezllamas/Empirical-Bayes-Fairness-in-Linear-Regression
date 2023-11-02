
%% Structure of the Code

% The code is composed of 4 parts. The first one prepares the data and
% modifies it into a usable form (Code_1.mat). The second one (Code_2.mat)
% separates the data into train and test sets. The third code (Code_3.mat) 
% calls the optimization problem and. The final code (Code_4.mat) uses the 
% saved solution to output the final graphs and save.


%% Code 3

function [Final_Solution] = Code_3(X,y,z,eps,NumT,a_ini,b_ini,beta0_ini,Sigma_ini)

[N,p] = size(X);
p = p-1;

% Log Marginal Likelihood

fun = @(x) aux_fun(x,X,y,z);

% Initial values

x_ini = zeros(p+1,4);
x_ini(1,1) = a_ini;
x_ini(1,2) = b_ini;
x_ini(2,1) = eps/2;
x_ini(:,3) = beta0_ini;
x_ini(:,4) = Sigma_ini;

% Lower bounds

LB = zeros(1,4*(p+1));
LB(1) = 1e-3;               % Lower bound for a
LB(p+2) = 1e-3;             % Lower bound for b
LB(2) = 0;                  % Lower bound for epsilon
LB(2*p+3:3*p+3) = -10^4;    % Lower bound for beta0_i
LB(3*p+4:end) = 10^-10;     % Lower bound for Sigma_ii 

% Upper bounds

UB = zeros(1,4*(p+1));
UB(1) = 1e5;                % Upper bound for a
UB(p+2) = 1e5;              % Upper bound for b
UB(2) = eps;                % Upper bound for epsilon
UB(2*p+3:3*p+3) = 10^4;     % Upper bound for beta0_i
UB(3*p+4:end) = 10^5;       % Upper bound for Sigma_ii 
            
% Optimization problem

problem = createOptimProblem('fmincon','objective', ...
          @(x) fun(x), 'x0', x_ini, 'lb', LB, 'ub', UB);
problem.options.MaxIterations=100000;
problem.options.MaxFunctionEvaluations=100000;
gs = GlobalSearch('Display','iter','StartPointsToRun','bounds-ineqs','MaxTime',NumT);
[Sol, f] = run(gs, problem);

% Solution

a_opt = Sol(1,1);               % Value of a at solution
b_opt = Sol(1,2);               % Value of b at solution
eps_opt = abs(Sol(2,1));        % Value of unfairness constraint at solution
beta0_opt = Sol(:,3);           % Value of beta0 at solution
Sigma_opt = diag(Sol(:,4));     % Value of Sigma at solution

Sigma1 = inv(X'*X+inv(Sigma_opt));                  % We calculate Sigma1          
beta1 = Sigma1*(X'*y + inv(Sigma_opt)*beta0_opt);   % We calculate beta1


if abs(z'*beta1)>=eps_opt       % We check that the fairness constraint holds
    beta1 = beta1 - (z'*beta1)*z/norm(z)^2 + z'\eps_opt;    % If it does not, then we force it
end

beta0_opt = Sigma_opt*(inv(Sigma1)*beta1-X'*y);     % We reevaluate beta0 in case beta1 changed.   


a1 = a_opt + N;
b1 = b_opt + y'*y + beta0_opt'*inv(Sigma_opt)*beta0_opt-beta1'*inv(Sigma1)*beta1;   % We do the same with b1.



Final_Solution = {Sol,a1,b1,beta1,Sigma1,eps_opt,f};


%% Auxiliary functions

function sol = restr(x,z,eps1)
    if abs(z'*x)<eps1    % If fairness constraint is fulfilled
        sol=x;
    else
        sol = x - (z'*x)*z/norm(z)^2 + z'\eps1;    % Else, project to a plane at which it is fulfilled with |zT beta1| = eps1
    end
end

function sol = aux_fun(x,X,y,z)
    a = x(1,1);
    b = x(1,2);
    eps0 = x(2,1);
    beta0 = x(:,3);
    Sigma = diag(x(:,4));

    [N_aux,p_aux] = size(X);

    Sigma1_aux = inv(X'*X+inv(Sigma));
   
    beta1_aux = Sigma1_aux*(X'*y+inv(Sigma)*beta0);

    if abs(z'*beta1_aux)>=abs(eps0)
        beta1_aux =beta1_aux - (z'*beta1_aux)*z/norm(z)^2 + z'\eps0;
    end

    beta0 = Sigma*(inv(Sigma1_aux)*beta1_aux-X'*y);        

    b1_aux = b + y'*y + beta0'*inv(Sigma)*beta0 - beta1_aux'*inv(Sigma1_aux)*beta1_aux;

    term1 = +1/2*log(det(eye(p_aux)+Sigma*(X'*X)));
    term2 = -a/2*log(b/2)+(a+N_aux)/2*log(b1_aux/2);
    term3 = -sum(log(a/2+(0:(floor(N_aux/2)-1))));

    sol=term1+term2+term3;      % Sum of terms of (log) marginal likelihood
end

end