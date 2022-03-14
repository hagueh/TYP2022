data3 = chickenpox_dataset;
data3 = [data3{:}];
data1 = readtable("extendedDataNew.xlsx");
data1 = data1(:,[2 3 4]);
data1 = table2array(data1);

%Just got tempData for now
data = zeros(1,276);
for i=1:length(data1)
    data(1,i) = data1(i,3);
end

%data = [data{:}];

figure
plot(data)
xlabel("Month")
ylabel("Max 2m Air Temperature")
title("Monthly Max 2m Air Temperatures in Spain 1997-2019")

%Partition training and test data 0.9/0.1
numTimeStepsTrain = floor(0.9*numel(data));

dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);

%% Standardise data
mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

%% Prepare Predictors and Responses
%The responses are the next item in the time series, so using the previous
%value to predict the next value
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

%% Define LSTM Network Architecture
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%Specify the training options
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

%% Train LSTM Network
net = trainNetwork(XTrain,YTrain,layers,options);

%% Forecast Future Time Steps
%Standardize the test data with the same parameters as the training data
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

%Make the first prediction using the last time step of the training
%response YTrain(end)
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

%Unstandardize the predictions with same parameters
YPred = sig*YPred + mu;

%Calculate the root-mean-square-error
YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2))

%Plot the data
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast")
legend(["Observed" "Forecast"])

%Compare forecasted values with the actual test data
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)