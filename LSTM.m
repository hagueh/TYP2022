

%% Prepare Training/Testing Data
data1 = readtable("extendedDataNew.xlsx");
data1 = data1(:,[2 3 4 5]);
data1 = table2array(data1);

%Just got tempData for now
data = zeros(4,276);
for i=1:length(data1)
    data(1,i) = data1(i,1);
    data(2,i) = data1(i,2);
    data(3,i) = data1(i,3);
    data(4,i) = data1(i,4);
end

%Split train/test 0.9/0.1
dataTrain = data(:,1:floor(0.9*276)+1);
dataTest = data(:,floor(0.9*276)+1:end);

%Training
predictors = cell(1,1);
predictors{1} = dataTrain(2:4,:);
responses = cell(1,1);
responses{1} = dataTrain(1,:);

XTrain = predictors;
YTrain = responses;

%Testing
predictors = cell(1,1);
predictors{1} = dataTest(2:4,:);
responses = cell(1,1);
responses{1} = dataTest(1,:);

XTest = predictors;
YTest = responses;

numFeatures = size(XTrain{1},1)

%Normalize Training Predictors
mu = mean([XTrain{:}],2);
sig = std([XTrain{:}],0,2);

for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - mu) ./ sig;
end


%Normalize Testing Predictors
mu = mean([XTest{:}],2);
sig = std([XTest{:}],0,2);
for i = 1:numel(XTest)
    XTest{i} = (XTest{i} - mu) ./ sig;
end


%Choose a mini-batch size which divides the training data evenly and reduces the amount of padding in the mini-batches
miniBatchSize = 1;

%% Define Network Architecture
%Create an LSTM network that consists of an LSTM layer with 200 hidden
% units, followed by a fully connected layer of size 50 and a dropout layer
% with dropout probability 0.5
numResponses = size(YTrain{1},1);
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%Specify the training options. Train for 60 epochs with mini-batches of
% size 20 using the solver 'adam'. Specify the learning rate 0.01. To
% prevent the gradients from exploding, set the gradient threshold to 1. To
% keep the sequences sorted by length, set 'Shuffle' to 'never'.
maxEpochs = 20;
miniBatchSize = 1;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Plots','training-progress',...
    'Verbose',0);

%% Train the network
net = trainNetwork(XTrain,YTrain,layers,options);

%% Make Prediction! batchsize 1 so theres no padding
YPred = predict(net,XTest,'MiniBatchSize',1);

 
%% Visualise
idx = randperm(numel(YPred),1);
figure
plot(YTest{idx(i)},'--')
hold on
plot(YPred{idx(i)},'.-')
hold off
xlabel("Time Step")
ylabel("RUL")


%RMSE
for i = 1:numel(YTest)
    YTestLast(i) = YTest{i}(end);
    YPredLast(i) = YPred{i}(end);
end
rmse = sqrt(mean((YPredLast - YTestLast).^2))





% %% NOT WORKING
% %% Generate The parameters with Time Series Forecasting using LSTM Deep learning Network
% 
%     data1 = readtable("extendedDataNew.xlsx");
%     data1 = data1(:,[2 3 4]);
%     data1 = table2array(data1);
% 
%     %Just got tempData for now
%     data = zeros(1,276);
%     for i=1:length(data1)
%         data(1,i) = data1(i,2);
%     end
%     forcasted1 = timeSeriesForcasting(data);
% 
%     data = zeros(1,276);
%     for i=1:length(data1)
%         data(1,i) = data1(i,3);
%     end
%     forcasted2 = timeSeriesForcasting(data);
% 
%     forcastNewData = zeros(2,61);
%     for i=1:61
%         forcastNewData(1,i) = forcasted1(1,i);
%         forcastNewData(2,i) = forcasted2(1,i);
%     end
% 
%     predictors = cell(1,1);
%     predictors{1} = forcastNewData(1:2,:);
%     Xforcast = predictors;
% 
%     %Normalize Training Predictors
%     mu = mean([XTrain{:}],2);
%     sig = std([XTrain{:}],0,2);
% 
%     for i = 1:numel(Xforcast)
%         Xforcast{i} = (Xforcast{i} - mu) ./ sig;
%     end
%     
%     %% Predict the future!
%     YPred = predict(net,Xforcast,'MiniBatchSize',1);
% 
%     %Visualise forecast
% 
%     data1 = readtable("extendedDataNew.xlsx");
%     data1 = data1(:,[2 3 4]);
%     data1 = table2array(data1);
% 
%     %Just got tempData for now
%     data = zeros(1,276);
%     for i=1:length(data1)
%         data(1,i) = data1(i,1);
%     end
% 
%     %Plot the data
%     figure
%     plot(data(1,:))
%     hold on
%     idx = 276:(276+length(YPred));
%     plot(idx,[data(276) YPred],'.-')
%     hold off
%     xlabel("Month")
%     ylabel("Cases")
%     title("2m Air Temperature Spain Forecast for next 5 years!!")
%     legend(["Observed" "Forecast"])





    function [futureForecast] = timeSeriesForcasting(data)
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

    dataAllStandardized = (data - mu) / sig;

    %% Prepare Predictors and Responses
    %The responses are the next item in the time series, so using the previous
    %value to predict the next value
    XTrain = dataTrainStandardized(1:end-1);
    YTrain = dataTrainStandardized(2:end);

    XdataAll = dataAllStandardized(1:end-1);
    YdataAll = dataAllStandardized(2:end);

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

    %% Predict actual future values 5 years (60 months)

    %Make the first prediction using the last time step of the test
    %response YTest(end)
    net = predictAndUpdateState(net,XdataAll);
    [net,YPred] = predictAndUpdateState(net,YdataAll(end));

    numTimeStepsTest = 60;
    for i = 2:numTimeStepsTest
        [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
    end

    %Unstandardize the predictions with same parameters
    YPred = sig*YPred + mu;

    %Plot the data
    figure
    plot(data(1:end-1))
    hold on
    idx = 276:(276+numTimeStepsTest);
    plot(idx,[data(276) YPred],'.-')
    hold off
    xlabel("Month")
    ylabel("Cases")
    title("Parameter Forecast for next 5 years")
    legend(["Observed" "Forecast"])

    futureForecast = [data(276) YPred];
end

