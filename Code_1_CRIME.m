%% Structure of the Code

% The code is composed of 4 parts. The first one prepares the data and
% modifies it into a usable form (Code_1.mat). The second one (Code_2.mat)
% separates the data into train and test sets. The third code (Code_3.mat) 
% calls the optimization problem and. The final code (Code_4.mat) uses the 
% saved solution to output the final graphs and save.


%% Code 1: Communities & CRIME


dat=load('dat_comm.mat');
data=dat.dat_comm;

y0=data(:,end);
X0=data(:,1:end-1);

perWhite=X0(:,4);
X0(:,4)=[];

[N,p]=size(X0);
 
y0=(y0-mean(y0))/std(y0);

X0=[ones(N,1),X0];

X0=X0;
y0=y0;

%%

notWhite=zeros(N,1);
isnotSens=zeros(N,1);
for i=1:N
    if perWhite(i)<mean(perWhite) %Less white than on average
        notWhite(i)=1;
        isnotSens(i)=1;
    end
end


vecWhite=[];
vecNonWhite=[];
for i=1:N
    if notWhite(i)==0
        vecWhite=[vecWhite;X0(i,:)];
    else
        vecNonWhite=[vecNonWhite;X0(i,:)]; 
    end
end

medWhite=[];
medNonWhite=[];
for i=1:p+1
    medWhite=[medWhite,mean(vecWhite(:,i))];
    medNonWhite=[medNonWhite,mean(vecNonWhite(:,i))];
end

z0=medWhite-medNonWhite;
z0=z0'/norm(z0);