%% Structure of the Code

% The code is composed of 4 parts. The first one prepares the data and
% modifies it into a usable form (Code_1.mat). The second one (Code_2.mat)
% separates the data into train and test sets. The third code (Code_3.mat) 
% calls the optimization problem and. The final code (Code_4.mat) uses the 
% saved solution to output the final graphs and save.


%% Code 1: Simulations

N = 3000;   % Number of data points
p = 100;      % Number of predictors
state = rng;    % Seed
kappa = 3/5;    % Similarity between Sensitive and Non-Sensitive beta

% Normal-Gamma based synthetic data

% Gamma Distribution
a_Orig = 1;     % Original hyper-parameter a
b_Orig = 20;    % Original hyper-parameter b

GammaDistr = makedist('gamma','a',a_Orig/2,'b',2/b_Orig);
phi_Orig = random(GammaDistr,1);    % Original value of phi

% Normal Distribution
beta0_Orig = 4.*(rand(1,p+1)-1/2)';     % Original value of beta_0
Sigma_Orig = diag(5.*rand(1,p+1));      % Original value of Sigma

beta_Orig_Sens = mvnrnd(beta0_Orig,1/phi_Orig*Sigma_Orig,1)';
beta_Orig_NonSens = mvnrnd(kappa*beta0_Orig,1/phi_Orig*Sigma_Orig,1)';


% Sensitive and Non-Sensitive classes

k = 2/3;     % Proportion of Non-Sensitive population

N_NS = floor(N*k);
index = unique(floor(1+(N-1)*rand(1,N_NS)));
N_NS = length(index);       % Number of Non-Sensitive population

% Data Generation (X)

X0 = zeros(N,p);
isnotSens = zeros(N,1);


for i=1:p
    aux = 10.*randn(N,1);
    for j=1:N
        if any(index(:) == j)
            isnotSens(j) = 1;
            X0(j,i) = aux(j);
        else
            X0(j,i) = aux(j);
        end
    end
end

X0 = [ones(N,1),X0];
while det(X0'*X0) == Inf    % In case the matrix is too large, we reduce it.
    X0 = X0.*0.9;
end

X0 = det(X0'*X0)^(-1/(2*(p+1))).*X0; % We normalize so that the determinant is not intractable

% Obtaining the synthetic data

epsilon = mvnrnd(zeros(1,N),(1/phi_Orig)*eye(N),1)';    % We generate random errors

y0 = zeros(N,1);

for i=1:N
    if isnotSens(i) == 1
        y0(i) = X0(i,:)*beta_Orig_NonSens + epsilon(i);     % We generate the data from the NG model for the non-sensitive class
    else
        y0(i) = X0(i,:)*beta_Orig_Sens + epsilon(i);        % We generate the data from the NG model for the sensitive class
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