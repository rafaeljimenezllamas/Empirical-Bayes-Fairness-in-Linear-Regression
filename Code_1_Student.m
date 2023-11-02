%% Structure of the Code

% The code is composed of 4 parts. The first one prepares the data and
% modifies it into a usable form (Code_1.mat). The second one (Code_2.mat)
% separates the data into train and test sets. The third code (Code_3.mat) 
% calls the optimization problem and. The final code (Code_4.mat) uses the 
% saved solution to output the final graphs and save.

%% Code 1: Student

T=readtable('student-por.csv');     % We read the data table

school=string(table2array(T(:,1)));
sex=string(table2array(T(:,2)));
age=table2array(T(:,3));
address=string(table2array(T(:,4)));
famsize=string(table2array(T(:,5)));
Pstatus=string(table2array(T(:,6)));
Medu=table2array(T(:,7));
Fedu=table2array(T(:,8));
traveltime=table2array(T(:,13));
studytime=table2array(T(:,14));
failures=table2array(T(:,15));
schoolsup=string(table2array(T(:,16)));
famsup=string(table2array(T(:,17)));
paid=string(table2array(T(:,18)));
activities=string(table2array(T(:,19)));
nursery=string(table2array(T(:,20)));
higher=string(table2array(T(:,21)));
internet=string(table2array(T(:,22)));
romantic=string(table2array(T(:,23)));
famrel=table2array(T(:,24));
freetime=table2array(T(:,25));
goout=table2array(T(:,26));
Dalc=table2array(T(:,27));
Walc=table2array(T(:,28));
health=table2array(T(:,29));
abscenses=table2array(T(:,30));
G1=table2array(T(:,31));
G2=table2array(T(:,32));
G3=table2array(T(:,33));

N=length(sex);
index=1:N;

p=25;

%%

X=zeros(N,p);

X(:,3)=age;
X(:,7)=Medu;
X(:,8)=Fedu;
X(:,9)=traveltime;
X(:,10)=studytime;
X(:,11)=failures;
X(:,19)=famrel;
X(:,20)=freetime;
X(:,21)=goout;
X(:,22)=Dalc;
X(:,23)=Walc;
X(:,24)=health;
X(:,25)=abscenses;

for i=1:N
    if school(i)=="GP"
        X(i,1)=1;
    end

    if sex(i)=="F"
        X(i,2)=1;
    end

    if address(i)=="U"
        X(i,4)=1;
    end

    if famsize(i)=="LE3"
        X(i,5)=1;
    end

    if Pstatus(i)=="T"
        X(i,6)=1;
    end

    if schoolsup(i)=="yes"
        X(i,12)=1;
    end

    if famsup(i)=="yes"
        X(i,13)=1;
    end

    if paid(i)=="yes"
        X(i,14)=1;
    end

    if activities(i)=="yes"
        X(i,15)=1;
    end

    if nursery(i)=="yes"
        X(i,16)=1;
    end

    if higher(i)=="yes"
        X(i,17)=1;
    end

%     if internet(i)=="yes"     % Our sensitive variable is access to the
                                % internet, so we delete it
%         X(i,18)=1;
%     end

    if romantic(i)=="yes"
        X(i,18)=1;
    end
end


X0=[ones(N,1),X];

X0 = det(X0'*X0)^(-1/(2*(p+1))).*X0; % We normalize so that the determinant is not intractable

y0=G3;

isnotSens=zeros(N,1);
isSens=zeros(N,1);

for i=1:N
    if internet(i)=="yes"
        isnotSens(i)=1;
    else
        isSens(i)=1;
    end
end


i=1;
while min(y0)<3             % We discard data points which are outliers.
    if y0(i)<3
        X0(i,:)=[];
        y0(i)=[];
        isnotSens(i)=[];
        isSens(i)=[];
        i=i-1;
    end
    i=i+1;
end
        

y0=(y0-mean(y0))/std(y0);       % We normalize y

N=length(y0);

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