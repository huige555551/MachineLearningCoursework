function neural_network(training_file, test_file, layers, units_per_layer, rounds)
    tFile = importdata(training_file);
    testFile = importdata(test_file);
    L = str2num(layers);
    U = str2num(units_per_layer);
    iterations = str2num(rounds);
   
    [m,n] = size(tFile);
    % normalizing the file values with respect to maximum value in all the attributes
    normalizingC = max(max(tFile(:,1:n-1)));
    tFile(:,1:n-1) = tFile(:,1:n-1)/normalizingC;
    
    classes = unique(tFile(:,n));
    [cm,~] = size(classes);
    
    xj=ones(m,n); % initializing xj for each line
    for i=1:m
        xj(i,2:end)=tFile(i,1:n-1);
    end  
    
    %initializing Input Layers and storing intial weights
    if L >2
        %hidden Layer exists 
        Wi=zeros(n,U,cm);
        WHidden=zeros(U+1,cm);
        for c=1:cm
            Wi(:,:,c)=getWeightsHiddenLayer(n,U); % input layer weights
            WHidden(:,c)=getWeights(U+1); %hidden layer weights
        end
        ZHidden=ones(U+1,cm);
        %Back propogation Algorithm main loop for hidden layer
        % training loop runs for the specified number of iterations
        for r=1:iterations
            %learning rate for every iteration
            learningRate=0.98.^(r-1);
            for i=1:m % for every input xn
                Zi=xj(i,:)';
                %Running for every class
                for c=1:cm
                    tnj=0;
                    if classes(c) == tFile(i,n)
                        tnj=1;
                    end
                    for h=1:U
                        ZHidden(h+1,c)=getZ(Wi(:,h,c)'*Zi);
                    end
                    Zj=getZ(WHidden(:,c)'*ZHidden(:,c));
                    DeltaOutput = (Zj - tnj) * Zj*(1-Zj);
                    HiddenDelta=zeros(U+1,1);
                    %updating the weights in the Hidden Layer L-1
                    for h=1:U+1
                        WHidden(h,c) = WHidden(h,c) - (learningRate * DeltaOutput * ZHidden(h,c));
                        HiddenDelta(h,1) = WHidden(h,c) * DeltaOutput * ZHidden(h,c) * (1-ZHidden(h,c));
                    end
                    
                    for h=2:U+1
                        for p=1:n
                            Wi(p,h-1,c) = Wi(p,h-1,c) - (learningRate * HiddenDelta(h,1) * Zi(p,1));
                        end
                    end
                end
            end
        end
    else
        Wi=zeros(n,cm); 
        for c=1:cm
            Wi(:,c)=getWeights(n); % input layer weights
        end
        %Back propogation Algorithm main loop
        % training loop runs for the specified number of iterations
        for r=1:iterations
            %learning rate for every iteration
            learningRate=0.98.^(r-1);
            for i=1:m % for every input xn
                Zi=xj(i,:)';
                %Running for every class
                for c=1:cm
                    tnj=0;
                    if classes(c) == tFile(i,n)
                        tnj=1;
                    end
                    Zj=getZ(Wi(:,c)'*Zi);
                    DeltaOutput = (Zj - tnj) * Zj*(1-Zj);
                    for p=1:n
                        Wi(p,c) = Wi(p,c) - (learningRate * DeltaOutput * Zi(p,1));
                    end
                end
            end
        end
    end
    
    %output training weights
    %disp('Hidden Weights');
    %disp(WHidden);
    %disp('Input Weights');
    %disp(Wi);
    
    %classification
    [tm,tn]=size(testFile);
    % normalizing the file values with respect to maximum value in all the attributes
    normalizingC = max(max(testFile(:,1:tn-1)));
    testFile(:,1:tn-1) = testFile(:,1:tn-1)/normalizingC;
    classification_accuracy =0.0;
    xjTest=ones(tm,tn); % initializing xjTest for each line in test file
    for i=1:tm
        xjTest(i,2:end)=testFile(i,1:tn-1);
    end
    
    for i=1:tm
        Zi=xjTest(i,:)';
        actualClass = testFile(i,tn);
        pClass = 0;
        tempHidden = ones(U+1,1);
        temp = zeros(cm,1);
        for c=1:cm
            if L>2
                %Hidden layer exists
                for h=1:U
                    tempHidden(h+1,1)=getZ(Wi(:,h,c)'*Zi);
                end
                temp(c,1) = getZ(WHidden(:,c)'*tempHidden(:,1));
            else
                %No Hidden layer
                temp(c,1) = getZ(Wi(:,c)'*Zi);
            end
        end
        maxSig = max(temp);
        accuracy =0;
        ties = length(find(temp(:,1) == maxSig));
        if ties > 1
            tiedClasses = find(temp == maxSig);
            existsInTiedClasses = 0;
            if find(tiedClasses == actualClass)
                existsInTiedClasses = 1;
            end
            accuracy = 1*existsInTiedClasses/ties;
        else
            pClass = classes(temp == maxSig);
            if actualClass == pClass
                accuracy =1;
            end
        end
        classification_accuracy = classification_accuracy + accuracy;
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n',i-1,pClass,actualClass,accuracy);
    end
    fprintf('\nclassification accuracy=%6.4f\n', classification_accuracy/tm)
end

function w= getWeights(size)
    w=zeros(1,size);
    a=-0.05;
    b=0.05;
    for i=1:size
        w(1,i)=(b-a)*rand(1,1) + a;
    end
end

function w=getWeightsHiddenLayer(sizeR,sizeC)
    % get weights for hidden layer
    w=zeros(sizeR,sizeC,1);
    a=-0.05;
    b=0.05;
    for i=1:sizeR
        for j =1:sizeC
            w(i,j)=(b-a)*rand(1,1) + a;
        end
    end
end

function z = getZ(a)
    z=1/(1 + exp(-a));
end