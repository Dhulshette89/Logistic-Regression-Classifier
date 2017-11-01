clear all; close all; clc;
data = dlmread('corrupted_2class_iris_dataset.dat');
%Shuffling data to get a random set
shuffled_data=data(randperm(100),:);
%saving total data size to count
DataSize=size(data,1);
%Number of folds
TotalFolds=10;
I1=shuffled_data(:,1);
I2=shuffled_data(:,2);
I3=shuffled_data(:,3);
I4=shuffled_data(:,4);
% Normalizing the data column wise (feature wise)%
ScaledI1=(I1-min(I1(:))) ./ (max(I1(:)-min(I1(:))));
ScaledI2=(I2-min(I2(:))) ./ (max(I2(:)-min(I2(:))));
ScaledI3=(I3-min(I3(:))) ./ (max(I3(:)-min(I3(:))));
ScaledI4=(I4-min(I4(:))) ./ (max(I4(:)-min(I4(:))));

 scalaedData = [ScaledI1(:,1) ScaledI2(:,1) ScaledI3(:,1) ScaledI4(:,1) shuffled_data(:,5)];
for k=1:TotalFolds
    count=0;
    %take test and train data for k fold cross validation%
    if k==1
      training_data=scalaedData(11: DataSize,:);
    else
      training_data=scalaedData([1:(k*10)-10 (k*10)+1:DataSize],:) ;
    end
    testing_data=scalaedData((k*10)-9:(k*10),:);
    X=[repmat(1,length(training_data),1) training_data(:,1) training_data(:,2) training_data(:,3) training_data(:,4)];
    y=training_data(:,5);
    
    TestX=[repmat(1,length(testing_data),1) testing_data(:,1) testing_data(:,2) testing_data(:,3) testing_data(:,4)];
    TestY=testing_data(:,5);
    
    learning_rate=0.15;
    %Weight vector initialization
    w=[1;1;1;1;1];
    costJ=[];
    i=1;
    
    for a=1:1500
      tempw0=w(1,1)-(learning_rate*((1/100)*sum((sigmoid(X*w) -y).*X(:,1))));
      tempw1=w(2,1)-(learning_rate*((1/100)*sum((sigmoid(X*w) -y).*X(:,2))));
      tempw2=w(3,1)-(learning_rate*((1/100)*sum((sigmoid(X*w) -y).*X(:,3))));
      tempw3=w(4,1)-(learning_rate*((1/100)*sum((sigmoid(X*w) -y).*X(:,4))));
      tempw4=w(5,1)-(learning_rate*((1/100)*sum((sigmoid(X*w) -y).*X(:,5))));
      
      w(1,1)=tempw0;
      w(2,1)=tempw1;
      w(3,1)=tempw2;
      w(4,1)=tempw3;
      w(5,1)=tempw4;
      costJ=[costJ;zeros(1,1)];
      s=log(sigmoid(X*w));
      costJ(i)=-1*( transpose(y)*log(sigmoid(X*w)) + transpose(1 - y)*log(1 - sigmoid(X*w)));
      i=i+1;
      
    end
    for k1=1:10
  % Assuming 0.5 is the threshold and anything >=05 should be classified as label 1
      if((sigmoid(TestX(k1,:)*w))>=0.5)
        predicted=1;
      else
        predicted=0;
      end
      if predicted==TestY(k1,1)
       count=count+1;
      end
    end
   %Calculate accuracy%
    accuracyRes=(count/10);
    Accuracy(k)= accuracyRes;
 end
 
transpose(Accuracy)
x1=mean(Accuracy);
fprintf('Average Accuarcy =%f \n',x1)
plot(costJ);% Size J = 1500x1 
ylim([0 250]);
xlim([0 1500]);
set(gca,'xtick',[0, 500, 1000, 1500]); 
xlabel('Training iterations'); 
ylabel('Cost function J'); 
   