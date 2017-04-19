function knn_classify(train, test,kValue)
    tFile = importdata(train);
    testFile = importdata(test);
    
    k=str2num(kValue);
    
    [tData,meanDimensionWise, std] = normalizeAndReturnData(tFile);
    testData = normalizeAndReturnTestData(testFile,meanDimensionWise,std);
    [m,n]=size(tData);
    [tm,~]=size(testData);
    classificationAccuracy = 0.0;
    for i=1:tm
        row=testData(i,:);
        dist=zeros(m,2);
        actualClass = testData(i,n);
        for j=1:m
            tRow = tData(j,:);
            dist(j,2) = tRow(n);
            dist(j,1) = getEucledianDistance(tRow,row,n);
        end
        sortValuesDistanceWise = sortrows(dist);
        kSortedValues = sortValuesDistanceWise(1:k,:);
        
        [predictedClass,F]= mode(kSortedValues(:,2));
        accuracy=0;
        if F < k/2
            % a tie has occured
            if ismember(actualClass,kSortedValues(:,2))
                accuracy = accuracy + 1/k;
            end
        elseif predictedClass == actualClass
            accuracy = accuracy+1;
        end
        classificationAccuracy = classificationAccuracy + accuracy;
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n', i-1, predictedClass, actualClass, accuracy);
    end
    fprintf('classification accuracy=%6.4f\n',classificationAccuracy/tm);
end

function dist = getEucledianDistance(tRow,row,n)
    dist=0.0;
    for i=1:n-1
        dist = dist + (row(1,i) - tRow(1,i)).^2;
    end
    dist = dist.^(1/2);
end

function data = normalizeAndReturnTestData(file,meanDimension,std)
    [m,n]=size(file);
    data = file;
    %normalizeData
    for i=1:n-1
        for j=1:m
            data(j,i) = (file(j,i) - meanDimension(i,1))/std(i,1);
        end
    end
end

function [data,meanDimension,std] = normalizeAndReturnData(file)
    [m,n]=size(file);
    meanDimension = zeros(n-1,1);
    std = zeros(n-1,1);
    % calculate mean and std
    for i=1:n-1
        meanDimension(i,1) = mean(file(:,i));
        std(i,1)=0.0;
        for j=1:m
            std(i,1)=std(i,1) + (file(j,i)-meanDimension(i,1)) * (file(j,i)-meanDimension(i,1));
        end
        std(i,1)=(std(i,1)/m).^(1/2);
    end
    data = file;
    %normalizeData
    for i=1:n-1
        for j=1:m
            data(j,i) = (file(j,i) - meanDimension(i,1))/std(i,1);
        end
    end
end