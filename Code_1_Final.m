%% Structure of the Code

% The code is composed of 4 parts. The first one prepares the data and
% modifies it into a usable form (Code_1_GH.mat). The second one (Code_2_GH.mat)
% separates the data into train and test sets. The third code (Code_3_GH.mat) 
% calls the optimization problem and. The final code (Code_4_GH.mat or Code_4_CV_GH.mat) uses the 
% saved solution to output the final graphs and save.


%% Code 1: Simulations

N = 1000;   % Number of data points
p = 2;      % Number of predictors

% Normal-Gamma based synthetic data

% Gamma Distribution
a_Orig = 50;     % Original hyper-parameter a
b_Orig = 1;    % Original hyper-parameter b

GammaDistr = makedist('gamma','a',a_Orig/2,'b',2/b_Orig);
phi_Orig = random(GammaDistr,1);    % Original value of phi

% Normal Distribution
beta0_Orig = 4.*(rand(1,p+1)-1/2)';     % Original value of beta_0
Sigma_Orig = diag(5.*rand(1,p+1));      % Original value of Sigma

beta_Orig = mvnrnd(beta0_Orig,1/phi_Orig*Sigma_Orig,1)';

% Data Generation (X)

X0 = zeros(N,p);
isnotSens = zeros(N,1);

for i=1:p
    X0(:,i) = 10.*randn(N,1);
end

X0 = [ones(N,1),X0];
while det(X0'*X0) == Inf    % In case the matrix is too large, we reduce it.
    X0 = X0.*0.9;
end

X0 = det(X0'*X0)^(-1/(2*(p+1))).*X0; % We normalize so that the determinant is not intractable

% Obtaining the synthetic data

epsilon = mvnrnd(zeros(1,N),(1/phi_Orig)*eye(N),1)';    % We generate random errors

y0 = X0*beta_Orig+epsilon;

% Sensitive and Non-Sensitive classes

y0k=0;
for i=1:N
    if y0(i)>y0k
        isnotSens(i)=1;
    else
        isnotSens(i)=0;
    end
end



%% We find z 

medNotSens=[];
medSens=[];

for i=1:N
    if isnotSens(i)==1 
        medNotSens=[medNotSens;X0(i,2:end)];
    else
        medSens=[medSens;X0(i,2:end)];
    end
end

medNotSensF=zeros(1,length(X0(1,2:end)));
for i=1:length(medNotSensF)
    medNotSensF(i)=mean(medNotSens(:,i));
end

medSensF=zeros(1,length(X0(1,2:end)));
for i=1:length(medSensF)
    medSensF(i)=mean(medSens(:,i));
end

z0=[0;medNotSensF'-medSensF'];
z0=z0./norm(z0);