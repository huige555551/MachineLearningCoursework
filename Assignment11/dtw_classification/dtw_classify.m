function dtw_classify(training_file,testing_file)
    trainingStructure = parseFilesAndReturnStructure(training_file);
    testStructure = parseFilesAndReturnStructure(testing_file);
    [m,~,~] = size(trainingStructure);
    [tm,~,~] = size(testStructure);
    classification_accuracy = 0;
    for i=1:tm
        objectID = testStructure(i,1,1);
        minimumCost = zeros(m,1);
        for j=1:m
            M = trainingStructure(j,5,1);
            N = testStructure(i,5,1);
            
            X = zeros(M,2);
            Y = zeros(N,2);
            
            X(:,1) = trainingStructure(j,3,1:M); 
            X(:,2) = trainingStructure(j,4,1:M);
            Y(:,1) = testStructure(i,3,1:N);
            Y(:,2) = testStructure(i,4,1:N);
            
            % initialization steps
            C = zeros(M,N);
            C(1,1) = calculateCost(X(1,:),Y(1,:));
            for k = 2:M
                C(k,1) = C(k-1,1) + calculateCost(X(k,:),Y(1,:));
            end
            for l = 2:N
                C(1,l) = C(1,l-1) + calculateCost(X(1,:),Y(l,:));
            end
            
            % main loop
            for k = 2:M
                for l = 2:N
                    C(k,l) = min(min(C(k-1,l), C(k,l-1)) ,C(k-1,l-1)) + calculateCost(X(k,:),Y(l,:));
                end
            end
            minimumCost(j,1)=C(M,N);
        end
        actualClass = testStructure(i,2,1);
        temp = min(minimumCost);
        index = find(minimumCost == temp);
        predictedClass = trainingStructure(index,2,1);
        accuracy = 0;
        if actualClass == predictedClass
            accuracy = 1;
        end
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f, distance = %.2f\n',objectID, predictedClass, actualClass, accuracy, temp);
        classification_accuracy = classification_accuracy + accuracy;
    end
    fprintf('classification accuracy=%6.4f\n', classification_accuracy/tm);
end

function cost = calculateCost(X,Y) % X=[2 2] Y =[3 3]
    % will return Eucledian distance
    cost = ((Y(1,2)-X(1,2)).^2 + (Y(1,1)-X(1,1)).^2).^(1/2);
end

function mainStructure = parseFilesAndReturnStructure(file)
    fileID = fopen(file);
    dHFFound = 0;
    k=0;
    while ~feof(fileID)
        fgetl(fileID);
        if contains(fgetl(fileID),'object ID')
            k=k+1;
        end
    end
    
    fclose(fileID);
    
    fileID = fopen(file);
    mainStructure = zeros(k,5);
    k=0;
    dimensionCounter=0;
    while ~feof(fileID)
        lineTemp = fgetl(fileID);
        if contains(lineTemp,'-------------------------------------------------')
            dHFFound = 0;
            if k > 0
                mainStructure(k,5,1) = dimensionCounter;
            end
            % resetting the dimensionCounter as we begin a new object id.
            dimensionCounter = 0;
        end
        if dHFFound
            dimensionCounter = dimensionCounter + 1;
            temp = strsplit(lineTemp);
            mainStructure(k,3,dimensionCounter) = str2double(temp{1,2});
            mainStructure(k,4,dimensionCounter) = str2double(temp{1,3});
        end
        if contains(lineTemp,'object ID')
            objectId = strsplit(lineTemp, ': ');
            k = k + 1;
            mainStructure(k,1,1) = str2double(objectId{1,2});
        elseif contains(lineTemp, 'class label')
            class = strsplit(lineTemp, ': ');
            mainStructure(k,2,1) = str2double(class{1,2});
        elseif contains(lineTemp,'dominant hand trajectory:')
            dHFFound = 1;
        end
        
    end
    mainStructure(k,5,1) = dimensionCounter;
    fclose(fileID);
end