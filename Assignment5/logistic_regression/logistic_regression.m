function logistic_regression(trainingFile,degree,testingFile)
    tFile=convertClasses(trainingFile);
    [m,n]=size(tFile);
    testFile=convertClasses(testingFile);
    M=str2num(degree);
    wSize=n;
    if M == 2
        wSize=2*(n-1) + 1;
    end
    W=zeros(wSize,1);
    Sigmoid=computeSigmoid(tFile,M,wSize);
    terminateLoop = 0;
    count = 0;
    prevDiff = 1;
    y=computeY(W,wSize,Sigmoid);
    sigT= transpose(Sigmoid);
    t=tFile(:,n);
    error = 0;
    for i=1:m
        error = error + t(i,1)*log(y(i,1)) + (1-t(i,1))*log(1-y(i,1));
    end
    error = -1*error;
    % main loop for Iterative Reweighted Least Squares
    while terminateLoop == 0
        count = count+1;
        R=computeR(m,y);
        Wnew = W - pinv(sigT*R*Sigmoid)*sigT*(y -t);
        yNew = computeY(Wnew,wSize,Sigmoid);
        diff = sum(Wnew - W);
        errorNew = 0;
        for i=1:m
            errorNew = errorNew + t(i,1)*log(yNew(i,1)) + (1-t(i,1))*log(1-yNew(i,1));
        end
        errorNew = -1*errorNew;
        if abs(prevDiff - diff) < 0.001 || abs(error - errorNew) < 0.001
            terminateLoop=1;
            W=Wnew;
            y=yNew;
            error=errorNew;
        else
            W=Wnew;
            y=yNew;
            prevDiff = diff;
            error=errorNew;
        end
    end
    for i=1:wSize
        fprintf('w%d=%.4f\n',i-1,W(i,1));
    end
    %classification
    [tm,tn] = size(testFile);
    %Sigmoid for testFile
    SigmoidTest = computeSigmoid(testFile,M,wSize);
    yTest=computeY(W,wSize,SigmoidTest);
    classification_accuracy=0;
    for i=1:tm
        actualClass = testFile(i,tn);
        accuracy = 0;
        tiedClasses=0;
        probability = yTest(i,1);
        predictedClass=0;
        %Logic to determing probability
        if yTest(i,1) > 0.5
            predictedClass = 1;
        elseif yTest(i,1) < 0.5
            predictedClass = 0;
            probability = 1 - yTest(i,1);
        else
            tiedClasses =1;
            predictedClass = 1;
        end
        %accuracy calculation
        if tiedClasses ~= 0
            accuracy = 0.5;
        else
            if actualClass == predictedClass
                accuracy =1;
            else
                accuracy =0;
            end
        end
        %Final output for classification
        classification_accuracy=classification_accuracy+accuracy;
        fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',i-1, predictedClass, probability, actualClass, accuracy);
    end
    fprintf('classification accuracy=%6.4f\n', classification_accuracy/tm);
end

function S=computeSigmoid(file,M,wSize)
    [m,n] = size(file);
    S=ones(m,wSize);
    if M == 1
        for i=1:m
            for j=2:wSize
                S(i,j)=file(i,j-1);
            end
        end
    else
        for i=1:m
            count=2;
            for j=1:n-1
                S(i,count)=file(i,j);
                count=count+1;
                S(i,count)=file(i,j)*file(i,j);
                count=count+1;
            end
        end
    end
end

% Converts class labels to binary format 0 or 1
function tFile =  convertClasses(fileName)
    tFile = importdata(fileName);
    [m,n]=size(tFile);
    for i=1:m
        if tFile(i,n) ~= 1
            tFile(i,n) = 0;
        end
    end
end

%computing y
function y = computeY(W,wSize,Sigmoid)
    [m,n]=size(Sigmoid);
    wT=transpose(W);
    y=zeros(m,1);
    for i=1:m
        Sig=transpose(Sigmoid(i,:));
        y(i,1)=1/(1+exp(-wT*Sig));
    end
end

%computing R
function R = computeR(m,y)
    R=zeros(m,m);
    for i=1:m
        R(i,i)=y(i,1)*(1-y(i,1));
    end
end