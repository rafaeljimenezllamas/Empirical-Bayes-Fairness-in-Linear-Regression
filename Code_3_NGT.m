function [Final_Solution] = Code_3_NGT(X,y,z,eps,NumT,NumIter,NumFev,NumPoints,a_ini,b_ini,beta0_ini,invSigma_ini)

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
x_ini = [];
x_ini = [x_ini;a_ini];
x_ini = [x_ini;b_ini];
x_ini = [x_ini;beta0_ini];

for i=1:p+1
    for j=i:p+1
        x_ini=[x_ini;invSigma_ini(i,j)];
    end
end

% Lower bounds

LB(1) = 1e-3;            % Lower bound for a
LB(2) = 1e-3;            % Lower bound for b
LB(3:3+p) = -1e4;        % Lower bound for beta0_i

for i=1:p+1
    LB=[LB,1e-6];       % Lower bound for Sigma_ii
    for j=i+1:p+1
        LB=[LB,-1e4];   % Lower bound for Sigma_ij, i=/=j
    end
end

% Upper bounds

UB(1) = 1e10;                   % Upper bound for a
UB(2) = 1e10;                   % Upper bound for b
UB(3:3+p) = 1e4;                % Upper bound for beta0_i
UB(4+p:length(x_ini)) = 1e4;    % Upper bound for Sigma_ii 


% Optimization problem

problem = createOptimProblem('fmincon','objective', ...
          @(x) fun(x), 'x0', x_ini, 'lb', LB, 'ub', UB,'nonlcon',@nonlcon);
problem.options.MaxIterations=NumIter;
problem.options.MaxFunctionEvaluations=NumFev;
gs = GlobalSearch('Display','iter','StartPointsToRun','bounds-ineqs','MaxTime',NumT,'NumTrialPoints',NumPoints);
[Sol, f] = run(gs, problem);


% Solution

a_opt = Sol(1);               % Value of a at solution
b_opt = Sol(2);               % Value of b at solution
beta0_opt = Sol(3:3+p);           % Value of beta0 at solution
invSigma_opt = zeros(p+1);         % Value of Sigma at solution
k=1;
for i=1:p+1
    for j=i:p+1
        invSigma_opt(i,j)=Sol(p+3+k);
        k=k+1;
    end
end 

invSigma_opt=invSigma_opt'*invSigma_opt;


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

    a = x(1);
    b = x(2);

    [~,paux] = size(Xaux);
    paux = paux-1;


    beta0 = x(3:3+paux);

    invSigma=zeros(paux+1);
    k=1;
    for i=1:paux+1
        for j=i:paux+1
            invSigma(i,j)=x(paux+3+k);
            k=k+1;
        end
    end

    invSigma=invSigma'*invSigma;

   
    
    Sigma1_aux = inv(Xaux'*Xaux+invSigma);

    beta1_aux = Sigma1_aux*(Xaux'*yaux+invSigma*beta0);

    c1=abs(zaux'*beta1_aux)-epsaux;
    
    c2=min(eig(invSigma))-1e-2;

    c(1)=c1;
    c(2)=-c2;
    ceq=[];
end

function sol = aux_fun(x,X,y,z,eps)
    [N_aux,p_aux] = size(X);

    a = x(1);
    b = x(2);

    beta0 = x(3:3+p);

    invSigma=zeros(p+1);
    k=1;
    for i=1:p+1
        for j=i:p+1
            invSigma(i,j)=x(p+3+k);
            k=k+1;
        end
    end

    invSigma=invSigma'*invSigma;


    Sigma1_aux = inv(X'*X+invSigma);
   
    beta1_aux = Sigma1_aux*(X'*y+invSigma*beta0);

    b1_aux = b + y'*y + beta0'*invSigma*beta0 - beta1_aux'*inv(Sigma1_aux)*beta1_aux;

    term1 = +1/2*log(det(eye(p_aux)+inv(invSigma)*(X'*X)));
    term2 = -a/2*log(b/2)+(a+N_aux)/2*log(b1_aux/2);
    term3 = -sum(log(a/2+(0:(floor(N_aux/2)-1))));

    sol=term1+term2+term3;      % Sum of terms of (log) marginal likelihood

end

end