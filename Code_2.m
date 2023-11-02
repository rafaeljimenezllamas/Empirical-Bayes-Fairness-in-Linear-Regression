%% Structure of the Code

% The code is composed of 4 parts. The first one prepares the data and
% modifies it into a usable form (Code_1.mat). The second one (Code_2.mat)
% separates the data into train and test sets. The third code (Code_3.mat) 
% calls the optimization problem and. The final code (Code_4.mat) uses the 
% saved solution to output the final graphs and save.

%% Code 2

M = floor(N/5); % Size of Test Set

Ind_Test = unique(floor(1+(N-1)*rand(1,M)));    % It selects M random values between 1 and N and it keeps the ones not repeating

Ind_Train=[];
for i=1:N
    if not(any(Ind_Test(:)==i))
        Ind_Train=[Ind_Train,i];
    end
end

M=length(Ind_Train);

X=X0(Ind_Train,:);      % We define the train set data
XT=X0(Ind_Test,:);      % We define the test set data

y=y0(Ind_Train);
yT=y0(Ind_Test);

medNotSens=[];      % Not sensitive data from the train set
medSens=[];         % Sensitive data from the train set

medNotSensT=[];     % Not sensitive data from the test set
medSensT=[];        % Sensitive data from the test set

for i=1:N
    if any(Ind_Train(:)==i)
        if isnotSens(i)==1 
            medNotSens=[medNotSens;X0(i,2:end)];
        else
            medSens=[medSens;X0(i,2:end)];
        end
    else
        if isnotSens(i)==1 
            medNotSensT=[medNotSensT;X0(i,2:end)];
        else
            medSensT=[medSensT;X0(i,2:end)];
        end
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

z=[0;medNotSensF'-medSensF'];       % We define z for the train set

z=z/norm(z);

% Test

medNotSensFT=zeros(1,length(X0(1,2:end)));
for i=1:length(medNotSensFT)
    medNotSensFT(i)=mean(medNotSensT(:,i));
end

medSensFT=zeros(1,length(X0(1,2:end)));
for i=1:length(medSensFT)
    medSensFT(i)=mean(medSensT(:,i));
end

zT=[0;medNotSensFT'-medSensFT'];    % We define z for the test set

zT=zT/norm(zT);
