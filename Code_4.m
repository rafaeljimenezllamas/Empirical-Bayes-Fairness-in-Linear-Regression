%% Structure of the Code

% The code is composed of 4 parts. The first one prepares the data and
% modifies it into a usable form (Code_1.mat). The second one (Code_2.mat)
% separates the data into train and test sets. The third code (Code_3.mat) 
% calls the optimization problem and. The final code (Code_4.mat) uses the 
% saved solution to output the final graphs and save.

%% Code 4

clear

%Code_1_Student
%Code_1_CRIME
Code_1              % We uncomment the dataset we want to use.



NumT = 500;     % Time limit.
Nsim = 1000;    % Number of simulations.


VecEps = logspace(-2,2,10);     % We define the grid of epsilons.


%%

% We define the initial values of the hyper-parameters

Sigma_ini = 1e-2.*ones(1,p+1);
beta0_ini = ones(1,p+1);
a_ini = 1;
b_ini = 1;

% For each value of epsilon, we solve the optimization problem

for i=1:length(VecEps)
    % We repeat the simulation 3 times for each epsilon. This is because
    % since we are imposing a Time Limit, each time we run the simulation
    % we may find a different solution. To maximize our chance of finding a
    % good solution, we can repeat the optimization problem.

    for k=1:3   
        Sol = Code_3(X0,y0,z0,VecEps(i),NumT,a_ini,b_ini,beta0_ini,Sigma_ini); % The output has the form {Sol,a1,b1,beta1,Sigma1,eps_opt,f};
        Mat = cell2mat(Sol(1));
        a1 = cell2mat(Sol(2));
        b1 = cell2mat(Sol(3));
        beta1 = cell2mat(Sol(4));
        Sigma1 = cell2mat(Sol(5));
        MargLik = cell2mat(Sol(end));

        a_opt = Mat(1,1);
        b_opt = Mat(1,2);
        eps_opt = abs(Mat(2,1));
        beta0_opt = Mat(:,3);
        Sigma_opt_aux=Mat(:,4);
        Sigma_opt = diag(Mat(:,4));

        % We find the predictive distribution

        A = eye(N)-X0*inv(X0'*X0+X0'*X0+Sigma_opt)*X0';
        B = X0*inv(X0'*X0+X0'*X0+Sigma_opt)*(inv(Sigma1)*beta1);
        C = b_opt + y0'*y0 + beta0_opt'*(inv(Sigma1)*beta1-X0'*y0) - (inv(Sigma1)*beta1)'*inv(X0'*X0+X0'*X0+Sigma_opt)*(inv(Sigma1)*beta1);
    
        mu_Train = X0*beta1;
        Var_Train = a1*A*inv(C-B'*inv(A)*B);
        Var_Train=nearestSPD(Var_Train);                % We check that it is a semidefinite positive matrix.
        Var_Train=triu(Var_Train,1)'+triu(Var_Train);   % We impose that the matrix is symmetric.
    
        M_ypred=mvnrnd(mu_Train,Var_Train,Nsim)';       % We create Nsim extractions from the distribution
    
        for j=1:Nsim
            Pred_Error(j) = 1/N * norm(y0-M_ypred(:,j)).^2;     % We calculate the distribution of the predictive error.
        end

        Pred_Error_aux{k}=Pred_Error;
        Mean_Error(k) = mean(Pred_Error);       % We find E[D]
        Fairness(k) = abs(z0'*beta1);           % We find mu_U

        Mat_aux{k}=Mat;
        a1_aux{k}=a1;
        b1_aux{k}=b1;
        beta1_aux{k}=beta1;
        Sigma1_aux{k}=Sigma1;
        MargLik_aux{k}=MargLik;

        a_ini_aux{k}=a_opt;
        b_ini_aux{k}=b_opt;
        beta0_ini_aux{k}=beta0_opt;
        Sigma_ini_aux{k}=Sigma_opt_aux;

        
    end


    [m,k0]=min(Mean_Error);     % We select the solution with the least error of the three.

    Vec_Pred_Error{i} = Pred_Error_aux{k0};
                                  
    ErrorMed(i) = Mean_Error(k0);
    FairnessMed(i) = Fairness(k0);

    Mat_Def{i}=Mat_aux{k0};
    a1_Def{i}=a1_aux{k0};
    b1_Def{i}=b1_aux{k0};
    beta1_Def{i}=beta1_aux{k0};
    Sigma1_Def{i}=Sigma1_aux{k0};
    MargLik_Def{i}=MargLik_aux{k0};

    a_ini=a_ini_aux{k0};            % We use the optimal values for the next initial point
    b_ini=b_ini_aux{k0};
    beta0_ini=beta0_ini_aux{k0};
    Sigma_ini=Sigma_ini_aux{k0};


end


%% Plots

% Estimated Predictive Error

figure
semilogx(VecEps,ErrorMed,'b','linewidth',2)     
xlabel('Epsilon')
ylabel('Estimated Predictive Error')
legend('E(D)')
grid on
grid minor

% Unfairness Measure

figure
semilogx(VecEps,FairnessMed,'b','linewidth',2)
hold on
semilogx(VecEps,VecEps,'--k','linewidth',2)
xlabel('Epsilon')
ylabel('Unfairness Measure')
grid on
grid minor


%% Saving the data

save('Saved_Data_File_Name.mat')