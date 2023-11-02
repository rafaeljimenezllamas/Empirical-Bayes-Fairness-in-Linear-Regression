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

Num = 5;    % Number of folds

%%

% For each fold we run the simulation

for n=1:Num
    
    % First we call Code_2 to separate the data between train and test
    % sets.
    Code_2

    % We define initial values.

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
            % Here we choose between using z or zT in Code_3. If we want to
            % impose the fairness constraint in the train set, we input z,
            % if instead we wish to impose it on the test set, we input zT.

            Sol = Code_3(X,y,z,VecEps(i),NumT,a_ini,b_ini,beta0_ini,Sigma_ini); % The output has the form {Sol,a1,b1,beta1,Sigma1,eps_opt,f};
            %Sol = Code_3(X,y,zT,VecEps(i),NumT,a_ini,b_ini,beta0_ini,Sigma_ini); % The output has the form {Sol,a1,b1,beta1,Sigma1,eps_opt,f};
           
            Mat = cell2mat(Sol(1));
            a1 = cell2mat(Sol(2));
            b1 = cell2mat(Sol(3));
            beta1 = cell2mat(Sol(4));
            Sigma1 = cell2mat(Sol(5));
            MargLik = cell2mat(Sol(6));
            
    
            a_opt = Mat(1,1);
            b_opt = Mat(1,2);
            eps_opt = abs(Mat(2,1));
            beta0_opt = Mat(:,3);
            Sigma_opt_aux=Mat(:,4);
            Sigma_opt = diag(Mat(:,4));
        
            % Predictive distribution for the Train set
    
            A = eye(M)-X*inv(X'*X+X'*X+Sigma_opt)*X';
            B = X*inv(X'*X+X'*X+Sigma_opt)*(inv(Sigma1)*beta1);
            C = b_opt + y'*y + beta0_opt'*(inv(Sigma1)*beta1-X'*y) - (inv(Sigma1)*beta1)'*inv(X'*X+X'*X+Sigma_opt)*(inv(Sigma1)*beta1);
        
            mu_Train = X*beta1;
            Var_Train = a1*A*inv(C-B'*inv(A)*B);
            Var_Train=nearestSPD(Var_Train);                % We check that it is a semidefinite positive matrix.
            Var_Train=triu(Var_Train,1)'+triu(Var_Train);   % We impose that the matrix is symmetric.
        
            M_ypred_Train=mvnrnd(mu_Train,Var_Train,Nsim)'; % We create Nsim extractions from the distribution
        
            for j=1:Nsim
                Pred_Error_Train(j) = 1/M * norm(y-M_ypred_Train(:,j)).^2;   % We calculate the distribution of the predictive error.
            end
    
            Pred_Error_Train_aux{k}=Pred_Error_Train;
            Mean_Error_Train(k) = mean(Pred_Error_Train);   % We find E[D]
            Fairness_Train(k) = abs(z'*beta1);              % We find mu_U
    
            % Predictive distribution for the Test set
    
            A = eye(N-M)-XT*inv(XT'*XT+X'*X+Sigma_opt)*XT';
            B = XT*inv(XT'*XT+X'*X+Sigma_opt)*(inv(Sigma1)*beta1);
            C = b_opt + y'*y + beta0_opt'*(inv(Sigma1)*beta1-X'*y) - (inv(Sigma1)*beta1)'*inv(XT'*XT+X'*X+Sigma_opt)*(inv(Sigma1)*beta1);
        
            mu_Test = XT*beta1;
            Var_Test = a1*A*inv(C-B'*inv(A)*B);        
            Var_Test=triu(Var_Test,1)'+triu(Var_Test);
            Var_Test=nearestSPD(Var_Test);
            Var_Test=triu(Var_Test,1)'+triu(Var_Test);
        
            M_ypred_Test=mvnrnd(mu_Test,Var_Test,Nsim)';
    
        
            for j=1:Nsim
                Pred_Error_Test(j) = 1/(N-M) * norm(yT-M_ypred_Test(:,j)).^2;
            end
    
            Pred_Error_Test_aux{k}=Pred_Error_Test;
            Mean_Error_Test(k) = mean(Pred_Error_Test);
            Fairness_Test(k) = abs(zT'*beta1);
    
    
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
    
        [m,k0]=min(Mean_Error_Train);
    
        Vec_Pred_Error_Train{i} = Pred_Error_Train_aux{k0};
                                      
        ErrorMed_Train(i) = Mean_Error_Train(k0);
        FairnessMed_Train(i) = Fairness_Train(k0);
    
        Vec_Pred_Error_Test{i} = Pred_Error_Test_aux{k0};
                                      
        ErrorMed_Test(i) = Mean_Error_Test(k0);
        FairnessMed_Test(i) = Fairness_Test(k0);
    
        Mat_Def{i}=Mat_aux{k0};
        a1_Def{i}=a1_aux{k0};
        b1_Def{i}=b1_aux{k0};
        beta1_Def{i}=beta1_aux{k0};
        Sigma1_Def{i}=Sigma1_aux{k0};
        MargLik_Def{i}=MargLik_aux{k0};
    
        a_ini=a_ini_aux{k0};             % We use the optimal values for the next initial point
        b_ini=b_ini_aux{k0};
        beta0_ini=beta0_ini_aux{k0};
        Sigma_ini=Sigma_ini_aux{k0};
    
    
    end
    
    %% Plots

    % Estimated Predictive Error
    figure
    semilogx(VecEps,ErrorMed_Train,'r','linewidth',2)
    hold on
    semilogx(VecEps,ErrorMed_Test,'b','linewidth',2)
    xlabel('Epsilon')
    ylabel('Predictive Error')
    grid on
    grid minor
    legend('Train Set','Test Set')
    
    % Unfairness Measure

    figure
    semilogx(VecEps,FairnessMed_Train,'r','linewidth',2)
    hold on
    semilogx(VecEps,FairnessMed_Test,'b','linewidth',2)
    semilogx(VecEps,VecEps,'--k','linewidth',2)
    xlabel('Epsilon')
    ylabel('Predictive Error')
    grid on
    grid minor
    legend('Train Set','Test Set')
    
    pause(0.01)     % To plot instantly and not wait for the full loop to finish
    
    filename = sprintf('Save_data_File_Name_%d.mat', n);
    save(filename);

end

save('Save_data_File_Name.mat')

tEnd = toc(tStart)

%%

for n=1:Num
    clearvars -except Num n Error_CV_Train_aux Error_CV_Test_aux Fairness_CV_Train_aux Fairness_CV_Test_aux MargLik_CV_aux
    filename = sprintf('Save_data_File_Name_%d.mat', n);
    load(filename);
    for i=2:length(VecEps)-1
        if cell2mat(MargLik_Def(i)) < cell2mat(MargLik_Def(i-1))        % We check that the solution found is better as epsilon grows.
            ErrorMed_Train(i)=ErrorMed_Train(i-1);                      % Since as epsilon grows the search region grows, the solution
            ErrorMed_Test(i)=ErrorMed_Test(i-1);                        % must be better, if it isn't, we select the last value which 
            FairnessMed_Train(i)=FairnessMed_Train(i-1);                % has a smaller value of the objective function (since we are minimizing)
            FairnessMed_Test(i)=FairnessMed_Test(i-1);
            MargLik_Def(i)=MargLik_Def(i-1);
        end
    end
    Error_CV_Train_aux(:,n)=ErrorMed_Train';
    Error_CV_Test_aux(:,n)=ErrorMed_Test';
    Fairness_CV_Train_aux(:,n)=FairnessMed_Train';
    Fairness_CV_Test_aux(:,n)=FairnessMed_Test';
    MargLik_CV_aux(:,n)=cell2mat(MargLik_Def');
end

for i=1:length(VecEps)
    Error_CV_Train(i)=mean(Error_CV_Train_aux(i,:));
    Error_CV_Test(i)=mean(Error_CV_Test_aux(i,:));
    Fairness_CV_Train(i)=mean(Fairness_CV_Train_aux(i,:));
    Fairness_CV_Test(i)=mean(Fairness_CV_Test_aux(i,:));
    MargLik_CV(i)=mean(MargLik_CV_aux(i,:));
end

%%


figure
semilogx(VecEps,Error_CV_Train,'r','linewidth',2)
hold on
semilogx(VecEps,Error_CV_Test,'b','linewidth',2)
xlabel('Epsilon')
ylabel('Predictive Error')
grid on
grid minor
legend('Train Set','Test Set')
set(gca,'FontSize',14)

figure
semilogx(VecEps,Fairness_CV_Test,'r','linewidth',2)
hold on
semilogx(VecEps,Fairness_CV_Train,'b','linewidth',2)
semilogx(VecEps,VecEps,'--k','linewidth',2)
xlabel('Epsilon')
ylabel('Predictive Error')
grid on
grid minor
legend('Train Set','Test Set')
set(gca,'FontSize',14)

figure
loglog(VecEps,MargLik_CV,'k','linewidth',2)
hold on
xlabel('Epsilon')
ylabel('Log(Marginal Likelihood)')
grid on
grid minor
set(gca,'FontSize',14)


