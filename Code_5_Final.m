clear

rng('default')      % We save the seed
state = rng;


% We select the dataset to solve

Code_1_Final
%Code_1_CRIME
%Code_1_Student


M=floor(N/5);                           % Approximate number of data in test set.
Num = 5;                                % Number of Folds.
NumT = 10^4;                            % Max optimization time for each value of epsilon
NumIter=50000;                          % Max number of iterations.
NumFev=50000;                           % Max number of function evaluations.
Nsim = 1000;                            % Number of simulations.
NumPoints = 5000;                       % Number of initial points generated.
VecEps = [logspace(-2,2,10),10^8];      % We define the grid of epsilons and the case epsilon so large that the fairness constraint does not affect
F = 1;                                  % If F=1 then the fairness constraint is enforced in the train set and otherwise in the test set.


% We select the names to save each case (NGT_NGD means that the NGT case is
% initialized in the solution of the NGD case and the same for ZP)

NameNGD = sprintf('Results_NGD');
NameZP = sprintf('Results_ZP');
NameNGT_NGD = sprintf('Results_NGT_NGD');
NameNGT_ZP = sprintf('Results_NGT_ZP');


t0_NGD=tic;
t0_ZP=tic;
t0_NGT=tic;

% Initial values for the first epsilon.

a_ini=1;
b_ini=1;
beta0_ini=zeros(p+1,1);
invSigma_ini=ones(p+1,1);
g_ini=1;

% We save the initial data

save('Results_Global.mat')

for n=1:Num

    [X,y,z,XT,yT,zT]=Code_2_fun_CV(X0,y0,isnotSens,M,n);     % We create the train and test sets
  
    
    iter = sprintf('_CV_%d.mat', n);
    
    % We solve for the ZP case

    t0_ZP=tic;
    Code_4_fun_ZP_Final(X,y,z,XT,yT,zT,F,VecEps,NumT,Nsim,NumIter,NumFev,NumPoints,a_ini,b_ini,beta0_ini,g_ini,strcat(NameZP,iter))
    t1_ZP(n)=toc(t0_ZP)

    % %%%%

    % We solve for the NGD case

    t0_NGD=tic;
    Code_4_fun_NGD_Final(X,y,z,XT,yT,zT,F,VecEps,NumT,Nsim,NumIter,NumFev,NumPoints,a_ini,b_ini,beta0_ini,invSigma_ini,strcat(NameNGD,iter))
    t1_NGD(n)=toc(t0_NGD)

    % %%%%
    load(strcat(NameNGD,iter),'a_ini_Def','b_ini_Def','beta0_ini_Def','invSigma_ini_Def')

    % We solve the NGT case using the optimal solution of the NGD case

    t0_NGT_ZP=tic;
    Code_4_fun_NGT_Final(X,y,z,XT,yT,zT,F,VecEps,NumT,Nsim,NumIter,NumFev,NumPoints,a_ini_Def,b_ini_Def,beta0_ini_Def,invSigma_ini_Def,strcat(NameNGT_NGD,iter))
    t1_NGT_ZP(n)=toc(t0_NGT_ZP)

    % %%%%

    % We solve the NGT case using the optimal solution of the ZP case

    load(strcat(NameZP,iter),'a_ini_Def','b_ini_Def','beta0_ini_Def','g_ini_Def')

    clear invSigma_ini_Def
    for k=1:length(VecEps)
        invSigma_ini_Def{k}=1/g_ini_Def{k}.*X'*X;
    end

    t0_NGT_NGD=tic;
    Code_4_fun_NGT_Final(X,y,z,XT,yT,zT,F,VecEps,NumT,Nsim,NumIter,NumFev,NumPoints,a_ini_Def,b_ini_Def,beta0_ini_Def,invSigma_ini_Def,strcat(NameNGT_ZP,iter))
    t1_NGT_NGD(n)=toc(t0_NGT_NGD)

end

% We save the overall results

save('Results_Final.mat')

%% Plots


for n=1:Num
    clearvars -except Num n Error_CV_Train_aux p97_5_CV_Train_aux p2_5_CV_Train_aux Error_CV_Test_aux p97_5_CV_Test_aux p2_5_CV_Test_aux Fairness_CV_Train_aux Fairness_CV_Test_aux MargLik_CV_aux Error_CV_NF_Train_aux Error_CV_NF_Test_aux Fairness_CV_NF_Train_aux Fairness_CV_NF_Test_aux MargLik_CV_NF_aux
    filename = sprintf('Results_NGD_CV_%d.mat', n);
    load(filename);

    for i=1:length(VecEps)
        p97_5_Train(i) = prctile(Vec_Pred_Error_Train{i},97.5);
        p2_5_Train(i) = prctile(Vec_Pred_Error_Train{i},2.5);
        p97_5_Test(i) = prctile(Vec_Pred_Error_Test{i},97.5);
        p2_5_Test(i) = prctile(Vec_Pred_Error_Test{i},2.5);
    end

    Error_CV_Train_aux(:,n)=ErrorMed_Train';
    Error_CV_Test_aux(:,n)=ErrorMed_Test';
    Fairness_CV_Train_aux(:,n)=FairnessMed_Train';
    Fairness_CV_Test_aux(:,n)=FairnessMed_Test';
    MargLik_CV_aux(:,n)=cell2mat(MargLik_Def');

    p97_5_CV_Train_aux(:,n)=p97_5_Train';
    p2_5_CV_Train_aux(:,n)=p2_5_Train';
    p97_5_CV_Test_aux(:,n)=p97_5_Test';
    p2_5_CV_Test_aux(:,n)=p2_5_Test';
end

for i=1:length(VecEps)
    Error_CV_Train(i)=mean(Error_CV_Train_aux(i,:));
    Error_CV_Test(i)=mean(Error_CV_Test_aux(i,:));
    Fairness_CV_Train(i)=mean(Fairness_CV_Train_aux(i,:));
    Fairness_CV_Test(i)=mean(Fairness_CV_Test_aux(i,:));
    MargLik_CV(i)=mean(MargLik_CV_aux(i,:));

    p97_5_CV_Train(i)=mean(p97_5_CV_Train_aux(i,:));
    p2_5_CV_Train(i)=mean(p2_5_CV_Train_aux(i,:));
    p97_5_CV_Test(i)=mean(p97_5_CV_Test_aux(i,:));
    p2_5_CV_Test(i)=mean(p2_5_CV_Test_aux(i,:));
end

% We save the final results for the Cross Validation for the NGD case

save('Results_Final_NGD_CV.mat')

%%

for n=1:Num
    clearvars -except Num n Error_CV_Train_aux p97_5_CV_Train_aux p2_5_CV_Train_aux Error_CV_Test_aux p97_5_CV_Test_aux p2_5_CV_Test_aux Fairness_CV_Train_aux Fairness_CV_Test_aux MargLik_CV_aux Error_CV_NF_Train_aux Error_CV_NF_Test_aux Fairness_CV_NF_Train_aux Fairness_CV_NF_Test_aux MargLik_CV_NF_aux
    filename = sprintf('Results_ZP_CV_%d.mat', n); 
    load(filename);

    for i=1:length(VecEps)
        p97_5_Train(i) = prctile(Vec_Pred_Error_Train{i},97.5);
        p2_5_Train(i) = prctile(Vec_Pred_Error_Train{i},2.5);
        p97_5_Test(i) = prctile(Vec_Pred_Error_Test{i},97.5);
        p2_5_Test(i) = prctile(Vec_Pred_Error_Test{i},2.5);
    end

    Error_CV_Train_aux(:,n)=ErrorMed_Train';
    Error_CV_Test_aux(:,n)=ErrorMed_Test';
    Fairness_CV_Train_aux(:,n)=FairnessMed_Train';
    Fairness_CV_Test_aux(:,n)=FairnessMed_Test';
    MargLik_CV_aux(:,n)=cell2mat(MargLik_Def');

    p97_5_CV_Train_aux(:,n)=p97_5_Train';
    p2_5_CV_Train_aux(:,n)=p2_5_Train';
    p97_5_CV_Test_aux(:,n)=p97_5_Test';
    p2_5_CV_Test_aux(:,n)=p2_5_Test';
end

for i=1:length(VecEps)
    Error_CV_Train(i)=mean(Error_CV_Train_aux(i,:));
    Error_CV_Test(i)=mean(Error_CV_Test_aux(i,:));
    Fairness_CV_Train(i)=mean(Fairness_CV_Train_aux(i,:));
    Fairness_CV_Test(i)=mean(Fairness_CV_Test_aux(i,:));
    MargLik_CV(i)=mean(MargLik_CV_aux(i,:));

    p97_5_CV_Train(i)=mean(p97_5_CV_Train_aux(i,:));
    p2_5_CV_Train(i)=mean(p2_5_CV_Train_aux(i,:));
    p97_5_CV_Test(i)=mean(p97_5_CV_Test_aux(i,:));
    p2_5_CV_Test(i)=mean(p2_5_CV_Test_aux(i,:));
end

% We save the final results for the Cross Validation for the ZP case

save('Results_Final_ZP_CV.mat')


%%

% We do the same for the NGT keeping the best solution from either the one
% found from the NGD solution or from the ZP solution

for n=1:Num
    clearvars -except Num n Error_CV_Train_aux p97_5_CV_Train_aux p2_5_CV_Train_aux Error_CV_Test_aux p97_5_CV_Test_aux p2_5_CV_Test_aux Fairness_CV_Train_aux Fairness_CV_Test_aux MargLik_CV_aux Error_CV_NF_Train_aux Error_CV_NF_Test_aux Fairness_CV_NF_Train_aux Fairness_CV_NF_Test_aux MargLik_CV_NF_aux
    filename = sprintf('Results_NGT_NGD_CV_%d.mat', n);
    load(filename);
    ErrorMed_Train_NGD = ErrorMed_Train;
    FairnessMed_Train_NGD = FairnessMed_Train;
    ErrorMed_Test_NGD = ErrorMed_Test;
    FairnessMed_Test_NGD = FairnessMed_Test;
    MargLik_Def_NGD = cell2mat(MargLik_Def);
    a1_Def_NGD = a1_Def;
    b1_Def_NGD = b1_Def;
    beta1_Def_NGD = beta1_Def;
    Sigma1_Def_NGD = Sigma1_Def;

    for i=1:length(VecEps)
        p97_5_Train_NGD(i) = prctile(Vec_Pred_Error_Train{i},97.5);
        p2_5_Train_NGD(i) = prctile(Vec_Pred_Error_Train{i},2.5);
        p97_5_Test_NGD(i) = prctile(Vec_Pred_Error_Test{i},97.5);
        p2_5_Test_NGD(i) = prctile(Vec_Pred_Error_Test{i},2.5);
    end
    
    filename = sprintf('Results_NGT_ZP_CV_%d.mat', n);
    load(filename);
    MargLik_Def_ZP = cell2mat(MargLik_Def);
    for i=1:length(VecEps)    
        if MargLik_Def_ZP(i) < MargLik_Def_NGD(i)
            MargLik_Def_NGT(i) = MargLik_Def_NGD(i);
            ErrorMed_Test_NGT(i) = ErrorMed_Test_NGD(i);
            FairnessMed_Test_NGT(i) = FairnessMed_Test_NGD(i);
            ErrorMed_Train_NGT(i) = ErrorMed_Train_NGD(i);
            FairnessMed_Train_NGT(i) = FairnessMed_Train_NGD(i);
            a1_Def{i} = a1_Def_NGD{i};
            b1_Def{i} = b1_Def_NGD{i};
            beta1_Def{i} = beta1_Def_NGD{i};
            Sigma1_Def{i} = Sigma1_Def_NGD{i};

            p97_5_Train(i) = p97_5_Train_NGD(i);
            p2_5_Train(i) = p2_5_Train_NGD(i);
            p97_5_Test(i) = p97_5_Test_NGD(i);
            p2_5_Test(i) = p2_5_Test_NGD(i);
        else
            MargLik_Def_NGT(i) = MargLik_Def_ZP(i);
            ErrorMed_Test_NGT(i) = ErrorMed_Test(i);
            FairnessMed_Test_NGT(i) = FairnessMed_Test(i);
            ErrorMed_Train_NGT(i) = ErrorMed_Train(i);
            FairnessMed_Train_NGT(i) = FairnessMed_Train(i);

            
            p97_5_Train(i) = prctile(Vec_Pred_Error_Train{i},97.5);
            p2_5_Train(i) = prctile(Vec_Pred_Error_Train{i},2.5);
            p97_5_Test(i) = prctile(Vec_Pred_Error_Test{i},97.5);
            p2_5_Test(i) = prctile(Vec_Pred_Error_Test{i},2.5);
        end
    end
    
    filename = sprintf('Results_NGT_CV_%d.mat', n);
    save(filename);

    p97_5_CV_Train_aux(:,n)=p97_5_Train';
    p2_5_CV_Train_aux(:,n)=p2_5_Train';
    p97_5_CV_Test_aux(:,n)=p97_5_Test';
    p2_5_CV_Test_aux(:,n)=p2_5_Test';

    Error_CV_Train_aux(:,n)=ErrorMed_Train';
    Error_CV_Test_aux(:,n)=ErrorMed_Test_NGT';
    Fairness_CV_Train_aux(:,n)=FairnessMed_Train';
    Fairness_CV_Test_aux(:,n)=FairnessMed_Test_NGT';
    MargLik_CV_aux(:,n)=MargLik_Def_NGT';
end

for i=1:length(VecEps)
    p97_5_CV_Train(i)=mean(p97_5_CV_Train_aux(i,:));
    p2_5_CV_Train(i)=mean(p2_5_CV_Train_aux(i,:));
    p97_5_CV_Test(i)=mean(p97_5_CV_Test_aux(i,:));
    p2_5_CV_Test(i)=mean(p2_5_CV_Test_aux(i,:));

    Error_CV_Train(i)=mean(Error_CV_Train_aux(i,:));
    Error_CV_Test(i)=mean(Error_CV_Test_aux(i,:));
    Fairness_CV_Train(i)=mean(Fairness_CV_Train_aux(i,:));
    Fairness_CV_Test(i)=mean(Fairness_CV_Test_aux(i,:));
    MargLik_CV(i)=mean(MargLik_CV_aux(i,:));
end

save('Results_Final_NGT_CV.mat')





%% Basic plots: E(D) with confidence intervals, E(D) and comparison with no Fairness, Fairness and Marginal Likelihood.

clear

load('Results_Final_NGT_CV.mat')     % We change it to whichever we want to graph                           

Lim=length(VecEps)-1;

figure
semilogx(VecEps(1:Lim),Error_CV_Train(1:Lim),'r','linewidth',2)
hold on
semilogx(VecEps(1:Lim),p97_5_CV_Train(1:Lim),'--r','linewidth',2)
semilogx(VecEps(1:Lim),Error_CV_Train(end)+0.*VecEps(1:Lim),'*-r','linewidth',2)
semilogx(VecEps(1:Lim),p2_5_CV_Train(1:Lim),'--r','linewidth',2)
semilogx(VecEps(1:Lim),Error_CV_Test(1:Lim),'b','linewidth',2)
semilogx(VecEps(1:Lim),p97_5_CV_Test(1:Lim),'--b','linewidth',2)
semilogx(VecEps(1:Lim),Error_CV_Test(end)+0.*VecEps(1:Lim),'*-b','linewidth',2)
semilogx(VecEps(1:Lim),p2_5_CV_Test(1:Lim),'--b','linewidth',2)
set(gca,'FontSize',14)
xlabel('$\varepsilon$','Interpreter','latex','FontSize',24)
ylabel('Estimated Predictive Error')
grid on
grid minor
legend('E(D) (train)','2.5th and 97.5th percentiles (train)','E(D) (train, $\varepsilon=\infty$)','','E(D) (test)','2.5th and 97.5th percentiles (test)','E(D) (test, $\varepsilon=\infty$)','Interpreter','latex','FontSize',12)



figure
semilogx(VecEps(1:Lim),Fairness_CV_Train(1:Lim),'r','linewidth',2)
hold on
semilogx(VecEps(1:Lim),Fairness_CV_Test(1:Lim),'b','linewidth',2)
semilogx(VecEps(1:Lim),Fairness_CV_Train(end)+0.*VecEps(1:Lim),'*-r','linewidth',2)
semilogx(VecEps(1:Lim),Fairness_CV_Test(end)+0.*VecEps(1:Lim),'*-b','linewidth',2)
semilogx(VecEps(1:Lim),VecEps(1:Lim),'--k','linewidth',2)
set(gca,'FontSize',14)
xlabel('$\varepsilon$','Interpreter','latex','FontSize',24)
ylabel('Unfairness measure')
grid on
grid minor
legend('$|\mu_U|$ (train)','$|\mu_U|$ (test)','$|\mu_U|$ (train, $\varepsilon=\infty$)','$|\mu_U|$ (test, $\varepsilon=\infty$)','$\varepsilon$','Interpreter','latex','FontSize',18)


figure (3)
semilogx(VecEps(1:Lim),MargLik_CV(1:Lim),'k','linewidth',2)
hold on
semilogx(VecEps(1:Lim),MargLik_CV(end)+.0*VecEps(1:Lim),'*-k','linewidth',2)
set(gca,'FontSize',14)
xlabel('$\varepsilon$','Interpreter','latex','FontSize',24)
ylabel('$f(a,b,\beta,\Sigma)$','Interpreter','latex','FontSize',24)
legend('$f(a,b,\beta,\Sigma)$','$f(a,b,\beta,\Sigma)$ ($\varepsilon=\infty$)','Interpreter','latex','FontSize',20)
grid on
grid minor