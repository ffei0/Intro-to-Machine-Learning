%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SVM
clear;
close all;
% generate data
num = 1000;
N = randn(num,2);
theta = unifrnd(-pi,pi,num,1);
rl = [2*ones(num/2,1);4*ones(num/2,1)];
data = [rl.*cos(theta) rl.*sin(theta)] + N;
labels = [-ones(num/2,1);ones(num/2,1)];

figure,title('Training Dataset'),hold on
plot(data(labels==-1,1),data(labels==1,2),'ro');
plot(data(labels==1,1),data(labels==1,2),'bo');
xlabel("x1");
ylabel("x2");

%%%
KSs = [0.1 0.5 1 1.5 2 2.5 3 3.5 4]; % box constraints parameter
BCs = [0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 2]; % Gaussian kernel width

for ii = 1:length(BCs)
    for jj = 1:length(KSs)
    fprintf('ii = %d / %d, jj = %d / %d ',ii, length(BCs),jj, length(KSs))    
    % SVM
    BC = BCs(ii); 
    KF = 'gaussian';
    KS = KSs(jj);
    PO = 3;
    OF = 0; 
    T=templateSVM( 'BoxConstraint',BC,'KernelFunction',KF,'KernelScale',KS);

    rand('seed',0);
    index = randperm(num);
    for fold=1:10
        foldwd = round(num/10);
        idx1 = (fold-1)*foldwd + 1;

        %%%  
        test_data = data([idx1:idx1+foldwd-1],:);
        test_label = labels([idx1:idx1+foldwd-1]);

        train_data = data;
        train_data([idx1:idx1+foldwd-1],:)=[];
        train_label = labels;
        train_label([idx1:idx1+foldwd-1],:)=[];

        % train svm
        svm = fitcecoc(train_data, train_label,'Learners',T); 
        [predict_label,predict_scores] = predict(svm, test_data);

        % training dataset accuracy
        accuracy = sum(predict_label == test_label)/numel(test_label); 
        fprintf('svm accuracy is %.02f \n',accuracy)
        acc_fold(fold) = accuracy;
    end
    accs(ii,jj) = mean(acc_fold);
    fprintf(' fold mean svm accuracy is %.02f \n',mean(acc_fold))
    end
end

figure, imagesc(accs), title(['SVM K-FOLD Hyperparameter Validation ' ...
    'Performance']), xlabel('Box Constraints'), ylabel(['Gaussian ' ...
    'Kernal Width'])
colorbar;

% generate test data
numT = 10000;
N = randn(numT,2);
theta = unifrnd(-pi,pi,numT,1);
rl = [2*ones(numT/2,1);4*ones(numT/2,1)];
dataT = [rl.*cos(theta) rl.*sin(theta)] + N;
labelsT = [-ones(numT/2,1);ones(numT/2,1)];

% select best svm model
[macc, midx] = max(accs(:));
[I,J] = ind2sub(length(BCs),length(KSs));
bestBC = BCs(I);
bestKS = KSs(J);
% 
BC = bestBC;
KF = 'gaussian'; 
KS = bestKS; 
PO = 3; 
OF = 0;

T=templateSVM( 'BoxConstraint',BC,'KernelFunction',KF,'KernelScale',KS);


% Train svm
svm = fitcecoc(dataT, labelsT,'Learners',T); 
[predict_label,predict_scores] = predict(svm, dataT); 

% Testing dataset accuracy
accuracy = sum(predict_label == labelsT)/numel(labelsT); 
fprintf('best BC = %f, best Ks = %f \n',bestBC, bestKS)
fprintf('svm accuracy is %.02f \n',accuracy)

figure,title('Testing Dataset'),hold on
plot(dataT(labelsT==-1,1),dataT(labelsT==-1,2),'ro');
plot(dataT(labelsT==1,1),dataT(labelsT==1,2),'bo');

figure,title('SVM Classification Performance'),hold on
plot(dataT(labelsT==1,1),dataT(labelsT==1,2),'bo');
plot(dataT(labelsT==-1,1),dataT(labelsT==-1,2),'bo');
plot(dataT(predict_label~=labelsT,1),dataT(predict_label~=labelsT,2),'ro')
xlabel("x1");
ylabel("x2");
legend('Correct Classification','','Incorrect Classification')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MLP
clear all;
close all;
clc;
% generate data
num = 1000;
N = randn(num,2);
theta = unifrnd(-pi,pi,num,1);
rl = [2*ones(num/2,1);4*ones(num/2,1)];
data = [rl.*cos(theta) rl.*sin(theta)] + N;
labels = [-ones(num/2,1);ones(num/2,1)];

figure,title('Training Dataset'),hold on
plot(data(labels==-1,1),data(labels==1,2),'ro');
plot(data(labels==1,1),data(labels==1,2),'bo');

labels(labels==-1)=2;

%%% 

numPerc = 15;
for i = 1:numPerc
    trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
    % Create a Pattern Recognition Network
    hiddenLayerSize = i;
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;
    net = patternnet(hiddenLayerSize);

    rand('seed',0);
    index = randperm(num);
    for fold=1:10
        foldwd = round(num/10);
        idx1 = (fold-1)*foldwd + 1;

        %
        test_data = data([idx1:idx1+foldwd-1],:);
        test_label = labels([idx1:idx1+foldwd-1]);
        
        train_data = data;
        train_data([idx1:idx1+foldwd-1],:)=[];
        train_label = labels;
        train_label([idx1:idx1+foldwd-1],:)=[];
        
        train_y = zeros(length(train_label),2);
        for tt=1:length(train_label)
            train_y(tt,train_label(tt)) = 1;
        end
        
        % train MLP
        [net,tr] = train(net,train_data',train_y');
        predict_label = net(test_data');
        [~,predict_label] = max(predict_label,[],1);
        predict_label = predict_label';

        % accuracy
        accuracy = sum(predict_label == test_label)/numel(test_label); 
        fprintf('mlp accuracy is %.02f \n',accuracy)
        acc_fold(fold) = accuracy;
    end
    accs(i) = mean(acc_fold);
    fprintf(' mean mlp accuracy is %.02f \n',mean(acc_fold))
end

figure;
plot(1:15,accs,'*')
xlabel('Number of Perceptrons')
ylabel('MLP Accuracy')
set(gca,'XTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]);
title('MLP K-Fold Hyperparameter Validation Performance')
% generate test data
num = 10000;
N = randn(num,2);
theta = unifrnd(-pi,pi,num,1);
rl = [2*ones(num/2,1);4*ones(num/2,1)];
data = [rl.*cos(theta) rl.*sin(theta)] + N;
labels = [-ones(num/2,1);ones(num/2,1)];
labels(labels==-1)=2;

test_y = zeros(length(labels),2);
for tt=1:length(labels)
    test_y(tt,labels(tt)) = 1;
end

% select best mlp model
[macc, midx] = max(accs(:));
bestLS = midx;
% 
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
% Create a Pattern Recognition Network
hiddenLayerSize = midx;
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;
net = patternnet(hiddenLayerSize);

% Train MLP
[net,tr] = train(net,data',test_y');
predict_label = net(data');
[~,predict_label] = max(predict_label,[],1); 
predict_label = predict_label';

%
accuracy = sum(predict_label == labels)/numel(labels); 
fprintf('best num of perceptrons = %f \n',bestLS)
fprintf('mlp accuracy is %.02f \n',accuracy)

figure,title('Testing Dataset'),hold on
plot(data(labels==2,1),data(labels==2,2),'ro');
plot(data(labels==1,1),data(labels==1,2),'bo');

figure,title('MLP Classification Performance'),hold on
plot(data(labels==2,1),data(labels==2,2),'bo');
plot(data(labels==1,1),data(labels==1,2),'bo');
plot(data(predict_label~=labels,1),data(predict_label~=labels,2),'ro')
xlabel("x1");
ylabel("x2");
legend('Correct Classiciation','','Inorrect Classiciation')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MLP
clear all;
close all;
clc;
% generate data
num = 1000;
N = randn(num,2);
theta = unifrnd(-pi,pi,num,1);
rl = [2*ones(num/2,1);4*ones(num/2,1)];
data = [rl.*cos(theta) rl.*sin(theta)] + N;
labels = [-ones(num/2,1);ones(num/2,1)];

figure,title('Training Dataset'),hold on
plot(data(labels==-1,1),data(labels==1,2),'ro');
plot(data(labels==1,1),data(labels==1,2),'bo');

labels(labels==-1)=2;

%%% 

numPerc = 15;
for i = 1:numPerc
    trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
    % Create a Pattern Recognition Network
    hiddenLayerSize = i;
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;
    net = patternnet(hiddenLayerSize);

    rand('seed',0);
    index = randperm(num);
    for fold=1:10
        foldwd = round(num/10);
        idx1 = (fold-1)*foldwd + 1;

        %
        test_data = data([idx1:idx1+foldwd-1],:);
        test_label = labels([idx1:idx1+foldwd-1]);
        
        train_data = data;
        train_data([idx1:idx1+foldwd-1],:)=[];
        train_label = labels;
        train_label([idx1:idx1+foldwd-1],:)=[];
        
        train_y = zeros(length(train_label),2);
        for tt=1:length(train_label)
            train_y(tt,train_label(tt)) = 1;
        end
        
        % train MLP
        [net,tr] = train(net,train_data',train_y');
        predict_label = net(test_data');
        [~,predict_label] = max(predict_label,[],1);
        predict_label = predict_label';

        % accuracy
        accuracy = sum(predict_label == test_label)/numel(test_label); 
        fprintf('mlp accuracy is %.02f \n',accuracy)
        acc_fold(fold) = accuracy;
    end
    accs(i) = mean(acc_fold);
    fprintf(' mean mlp accuracy is %.02f \n',mean(acc_fold))
end

figure;
plot(1:15,accs,'*')
xlabel('Number of Perceptrons')
ylabel('MLP Accuracy')
set(gca,'XTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]);
title('MLP K-Fold Hyperparameter Validation Performance')
% generate test data
num = 10000;
N = randn(num,2);
theta = unifrnd(-pi,pi,num,1);
rl = [2*ones(num/2,1);4*ones(num/2,1)];
data = [rl.*cos(theta) rl.*sin(theta)] + N;
labels = [-ones(num/2,1);ones(num/2,1)];
labels(labels==-1)=2;

test_y = zeros(length(labels),2);
for tt=1:length(labels)
    test_y(tt,labels(tt)) = 1;
end

% select best mlp model
[macc, midx] = max(accs(:));
bestLS = midx;
% 
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
% Create a Pattern Recognition Network
hiddenLayerSize = midx;
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;
net = patternnet(hiddenLayerSize);

% Train MLP
[net,tr] = train(net,data',test_y');
predict_label = net(data');
[~,predict_label] = max(predict_label,[],1); 
predict_label = predict_label';

%
accuracy = sum(predict_label == labels)/numel(labels); 
fprintf('best num of perceptrons = %f \n',bestLS)
fprintf('mlp accuracy is %.02f \n',accuracy)

figure,title('Testing Dataset'),hold on
plot(data(labels==2,1),data(labels==2,2),'ro');
plot(data(labels==1,1),data(labels==1,2),'bo');

figure,title('MLP Classification Performance'),hold on
plot(data(labels==2,1),data(labels==2,2),'bo');
plot(data(labels==1,1),data(labels==1,2),'bo');
plot(data(predict_label~=labels,1),data(predict_label~=labels,2),'ro')
xlabel("x1");
ylabel("x2");
legend('Correct Classiciation','','Inorrect Classiciation')



