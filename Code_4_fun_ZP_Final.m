
function [] = Code_4_fun_ZP_Final(X,y,z,XT,yT,zT,F,VecEps,NumT,Nsim,NumIter,NumFev,NumPoints,a_ini,b_ini,beta0_ini,g_ini,NameZP)

tStart = tic;

for i=1:length(VecEps)  

    if F==1
        Sol = Code_3_ZP(X,y,z,VecEps(i),NumT,NumIter,NumFev,NumPoints,a_ini,b_ini,beta0_ini,g_ini);
    else
        Sol = Code_3_ZP(X,y,zT,VecEps(i),NumT,NumIter,NumFev,NumPoints,a_ini,b_ini,beta0_ini,g_ini);
    end

    Mat = cell2mat(Sol(1));
    a1 = cell2mat(Sol(2));
    b1 = cell2mat(Sol(3));
    beta1 = cell2mat(Sol(4));
    Sigma1 = cell2mat(Sol(5));
    MargLik = cell2mat(Sol(6));  

    a_opt = Mat(1,1);
    b_opt = Mat(1,2);
    g_opt = Mat(2,1);
    beta0_opt = Mat(:,3);
    invSigma_opt = 1/g_opt*(X'*X);


    % Predictive distribution for the Train set

    [M,~]=size(X);
    [M_aux,~]=size(XT);
    N=M+M_aux;
    
    A = eye(M)-X*inv(X'*X+X'*X+invSigma_opt)*X';
            
    mu_Train = X*beta1;
    Var_Train = b1*inv(A)/a1;
    Var_Train=nearestSPD(Var_Train);                % We check that it is a semidefinite positive matrix.
    Var_Train=triu(Var_Train,1)'+triu(Var_Train);   % We impose that the matrix is symmetric.

    M_ypred_Train=mvnrnd(mu_Train,Var_Train,Nsim)'; % We create Nsim extractions from the distribution

    for j=1:Nsim
        Pred_Error_Train(j) = 1/M * norm(y-M_ypred_Train(:,j)).^2;   % We calculate the distribution of the predictive error.
    end

    % Predictive distribution for the Test set

    A = eye(N-M)-XT*inv(XT'*XT+X'*X+invSigma_opt)*XT';

    mu_Test = XT*beta1;
    Var_Test = b1*inv(A)/a1;        
    Var_Test=nearestSPD(Var_Test);
    Var_Test=triu(Var_Test,1)'+triu(Var_Test);

    M_ypred_Test=mvnrnd(mu_Test,Var_Test,Nsim)';


    for j=1:Nsim
        Pred_Error_Test(j) = 1/(N-M) * norm(yT-M_ypred_Test(:,j)).^2;
    end


    if i>1 && MargLik<MargLik_Def{i-1}      
            
        Vec_Pred_Error_Train{i} = Vec_Pred_Error_Train{i-1};                          
        ErrorMed_Train(i) = ErrorMed_Train(i-1);
        FairnessMed_Train(i) = FairnessMed_Train(i-1);

        Vec_Pred_Error_Test{i} = Vec_Pred_Error_Test{i-1};                          
        ErrorMed_Test(i) = ErrorMed_Test(i-1);
        FairnessMed_Test(i) = FairnessMed_Test(i-1);
    
        Mat_Def{i}=Mat_Def{i-1};
        a1_Def{i}=a1_Def{i-1};
        b1_Def{i}=b1_Def{i-1};
        beta1_Def{i}=beta1_Def{i-1};
        Sigma1_Def{i}=Sigma1_Def{i-1};
        MargLik_Def{i}=MargLik_Def{i-1};

        a_ini_Def{i}=a_ini_Def{i-1};
        b_ini_Def{i}=b_ini_Def{i-1};
        beta0_ini_Def{i}=beta0_ini_Def{i-1};
        g_ini_Def{i}=g_ini_Def{i-1};
    
        a_ini=a_ini_Def{i-1};             % We use the optimal values for the next initial point
        b_ini=b_ini_Def{i-1};
        beta0_ini=beta0_ini_Def{i-1};
        g_ini=g_ini_Def{i-1};

    else

        Vec_Pred_Error_Train{i} = Pred_Error_Train;                        
        ErrorMed_Train(i) = mean(Pred_Error_Train);
        FairnessMed_Train(i) = abs(z'*beta1);
    
        Vec_Pred_Error_Test{i} = Pred_Error_Test;               
        ErrorMed_Test(i) = mean(Pred_Error_Test);
        FairnessMed_Test(i) = abs(zT'*beta1);
    
        Mat_Def{i}=Mat;
        a1_Def{i}=a1;
        b1_Def{i}=b1;
        beta1_Def{i}=beta1;
        Sigma1_Def{i}=Sigma1;
        MargLik_Def{i}=MargLik;

        a_ini_Def{i}=a_opt;
        b_ini_Def{i}=b_opt;
        beta0_ini_Def{i}=beta0_opt;
        g_ini_Def{i}=g_opt;
    
        a_ini=a_opt;             % We use the optimal values for the next initial point
        b_ini=b_opt;
        beta0_ini=beta0_opt;
        g_ini=g_opt;

    end


end


for i=1:length(VecEps)
    p97_5_Train(i) = prctile(Vec_Pred_Error_Train{i},97.5);
    p2_5_Train(i) = prctile(Vec_Pred_Error_Train{i},2.5);
    p97_5_Test(i) = prctile(Vec_Pred_Error_Test{i},97.5);
    p2_5_Test(i) = prctile(Vec_Pred_Error_Test{i},2.5);
end


% We plot to see the result for each fold while the full simulation runs.

% Estimated Predictive Error

Lim=length(VecEps)-1;

figure
semilogx(VecEps(1:Lim),ErrorMed_Train(1:Lim),'r','linewidth',2)
hold on
semilogx(VecEps(1:Lim),ErrorMed_Test(1:Lim),'b','linewidth',2)
semilogx(VecEps(1:Lim),ErrorMed_Train(1:Lim)*0+ErrorMed_Train(end),'-*r','linewidth',2)
semilogx(VecEps(1:Lim),ErrorMed_Test(1:Lim)*0+ErrorMed_Test(end),'-*b','linewidth',2)
semilogx(VecEps(1:Lim),p97_5_Train(1:Lim),'--r','linewidth',2)
semilogx(VecEps(1:Lim),p2_5_Train(1:Lim),'--r','linewidth',2)
semilogx(VecEps(1:Lim),p97_5_Test(1:Lim),'--b','linewidth',2)
semilogx(VecEps(1:Lim),p2_5_Test(1:Lim),'--b','linewidth',2)
set(gca,'FontSize',14)
xlabel('$\varepsilon$','Interpreter','latex','FontSize',24)
ylabel('Estimated Predictive Error')
grid on
grid minor
legend('E(D) (train)','E(D) (test)')


% Unfairness Measure

figure
semilogx(VecEps(1:Lim),FairnessMed_Train(1:Lim),'r','linewidth',2)
hold on
semilogx(VecEps(1:Lim),FairnessMed_Test(1:Lim),'b','linewidth',2)
semilogx(VecEps(1:Lim),FairnessMed_Train(1:Lim)*0+FairnessMed_Train(end),'-*r','linewidth',2)
semilogx(VecEps(1:Lim),FairnessMed_Test(1:Lim)*0+FairnessMed_Test(end),'-*b','linewidth',2)
semilogx(VecEps(1:Lim),VecEps(1:Lim),'--k','linewidth',2)
set(gca,'FontSize',14)
xlabel('$\varepsilon$','Interpreter','latex','FontSize',24)
ylabel('Unfairness measure')
grid on
grid minor
legend('$|\mu_U|$ (train)','$|\mu_U|$ (test)','Interpreter','latex','FontSize',24)


% Marginal Likelihood

figure
semilogx(VecEps(1:Lim),cell2mat(MargLik_Def(1:Lim)),'k','linewidth',2)
hold on
semilogx(VecEps(1:Lim),VecEps(1:Lim)*0+cell2mat(MargLik_Def(end)),'-*k','linewidth',2)
set(gca,'FontSize',14)
xlabel('$\varepsilon$','Interpreter','latex','FontSize',24)
ylabel('$f(a,b,\beta,\Sigma)$','Interpreter','latex','FontSize',24)
grid on
grid minor

pause(0.01)     % To plot instantly and not wait for the full loop to finish

tIter = toc(tStart);

save(NameZP);   % We save the simulation

end
