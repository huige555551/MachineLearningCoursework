function naive_bayes(trainingFile,testingFile,classifierType,number)
    tline = importdata(trainingFile);
    testFile = importdata(testingFile);
    [m,n] = size(tline);
    classes= unique(tline(:,n));
    [cm,cn] = size(classes);
    classesLength = zeros(cm,1);
    for i=1:cm
        for j=1:m
            if(tline(j,n)==classes(i,1))
                classesLength(i,1)=classesLength(i,1)+1;
            end
        end
    end
    switch classifierType
        case 'histograms'
            histograms(tline,testFile,classes,number,classesLength);
        case 'gaussians'
            gaussians(tline,testFile,classes,classesLength);
        case 'mixtures'
            mixtures(tline,testFile,classes,number,classesLength);
        otherwise
            disp('Invalid choice!')
    end
end

function mixtures(tline,testFile,classes,number,classesLength)
[m,n] = size(tline);
    [cm,cn] = size(classes);
    Number = str2num(number);
    S=Inf;
    L=-Inf;
    LargeSmallG = zeros(cm*(n-1)*Number,8);
    rowStart=1;
    unset=0;
    for c=1:cm
        for i=1:n-1
            for j=1:m
                if classes(c,1) == tline(j,n)
                    % calculating the S and L for every class and storing
                    % it in the LargeSmallG matrix for later use
                    if unset==0
                        S = tline(j,i);
                        L = tline(j,i);
                        unset=1;
                    elseif tline(j,i) < S
                        S=tline(j,i);
                    elseif tline(j,i) > L
                        L=tline(j,i);
                    end
                end
            end
            for gn=1:Number
                LargeSmallG(rowStart,1)=classes(c,1); % class
                LargeSmallG(rowStart,2)=i-1; % dimension
                LargeSmallG(rowStart,3)=gn; % Gaussian Number
                G=(L-S)/(Number);
                LargeSmallG(rowStart,4)=G; % G
                LargeSmallG(rowStart,5)=S+(gn-1)*G+(G/2); % Setting the initial mean
                LargeSmallG(rowStart,6)=1; % Setting the standard deviation to 1
                LargeSmallG(rowStart,7)=1/Number; % setting the initial weight
                rowStart=rowStart+1;
            end
            unset=0;
        end
    end
    newPI= 3.1415926535897932384626;
    for i=1:cm
        for j=1:n-1
            temp=zeros(classesLength(i,1),1);
            cRow=1;
            for row=1:m
                % do this for every class
                if tline(row,n) == classes(i)
                    temp(cRow,1)=tline(row,j);
                    cRow=cRow+1;
                end
            end
            %storing the structure for every attribute value for all gaussian mixtures and denominator
            pij=zeros(classesLength(i,1),Number+1);
            
            % Now running the EM loop 50 times
            for em=1:50
                % We first do the E step which is computing pij
                finalRowNumber=zeros(Number,1);
                for mix=1:Number
                    for g=1:cm*(n-1)*Number
                        if LargeSmallG(g,1)==classes(i) && LargeSmallG(g,2)==(j-1) && LargeSmallG(g,3)==mix
                            finalRowNumber(mix,1)=g;
                        end
                    end
                end
                for mix=1:Number
                    meanMixture=LargeSmallG(finalRowNumber(mix,1),5);
                    stdMixture=LargeSmallG(finalRowNumber(mix,1),6);
                    weightMixture=LargeSmallG(finalRowNumber(mix,1),7);
                    for c=1:classesLength(i,1)
                        element=temp(c,1);
                        expFactor=exp(-1*(element-meanMixture)*(element-meanMixture)/(2*stdMixture*stdMixture));
                        Nixj=expFactor/(stdMixture*((2*newPI).^(1/2)));
                        pij(c,mix)=Nixj*weightMixture;
                        pij(c,Number+1)=pij(c,Number+1)+(Nixj*weightMixture);
                    end
                end
                %Finally using denominator calculated for all the mixtures
                %to update pij
                for mix=1:Number
                    for c=1:classesLength(i,1)
                        pij(c,mix)=pij(c,mix)/pij(c,Number+1);
                        LargeSmallG(finalRowNumber(mix,1),8)=pij(c,mix);
                    end
                end
                %now the m step where we update mean,weight,standard
                %deviation based on pij value
                wDenom=0;
                for mix=1:Number
                    for c=1:classesLength(i,1)
                        wDenom=wDenom+pij(c,mix);
                    end
                end
                for mix=1:Number
                    newWeightNumerator=0;
                    meanNumerator=0;
                    for c=1:classesLength(i,1)
                        newWeightNumerator=newWeightNumerator+pij(c,mix);
                        meanNumerator=meanNumerator+(pij(c,mix)*temp(c,1));
                        
                    end
                    LargeSmallG(finalRowNumber(mix,1),5)=meanNumerator/newWeightNumerator;
                    LargeSmallG(finalRowNumber(mix,1),7)=newWeightNumerator/wDenom;
                end
                for mix=1:Number
                    newMean=LargeSmallG(finalRowNumber(mix,1),5);
                    stdNumerator=0;
                    newWeightNumerator=0;
                    for c=1:classesLength(i,1)
                        newWeightNumerator=newWeightNumerator+pij(c,mix);
                        stdNumerator=stdNumerator + pij(c,mix)*(temp(c,1)-newMean)*(temp(c,1)-newMean);
                    end
                    newStd=(stdNumerator/newWeightNumerator).^(1/2);
                    if newStd < 0.01
                        newStd=0.01;
                    end
                    LargeSmallG(finalRowNumber(mix,1),6)=newStd;
                end
            end
        end
    end
    finalLoopEnd = cm*(n-1)*Number;
    for i=1:finalLoopEnd
        fprintf('Class %d, attribute %d, Gaussian %d, mean = %.2f, std = %.2f\n',LargeSmallG(i,1),LargeSmallG(i,2),LargeSmallG(i,3)-1,LargeSmallG(i,5),LargeSmallG(i,6));
    end
    
    %classification of mixtures of gaussians
    newPI= 3.1415926535897932384626;
    classification_accuracy=0;
    [tm,tn]=size(testFile);
    for i=1:tm
        classStructure = zeros(cm,2);
        actualClass=testFile(i,tn);
        classStructure(:,1)=classes(:,1); %storing all classes with their class numbers in this structure
        evidence=0;
        for c=1:cm
            for k=1:Number
                posteriorProb=classesLength(c)/m;
                for j=1:tn-1
                    for mainLoop=1:finalLoopEnd
                        if LargeSmallG(mainLoop,1)==classes(c) && LargeSmallG(mainLoop,2)==j-1 && k==LargeSmallG(mainLoop,3)
                            mean=LargeSmallG(mainLoop,5);
                            std=LargeSmallG(mainLoop,6);
                            expFactor = exp(-1*(testFile(i,j)-mean)*(testFile(i,j)-mean)/(2*std*std));
                            posteriorProb = posteriorProb*expFactor/((2*newPI*std*std).^(1/2));
                        end
                    end
                end
                if posteriorProb > classStructure(c,2)
                    evidence=evidence+posteriorProb;
                    classStructure(c,2)=posteriorProb;
                end
            end
        end
        accuracy=0.0;
        maxP=max(classStructure(:,2));
        predictedC=classStructure(classStructure(:,2)==maxP,1);
        tieClasses = length(find(classStructure(:,2)==maxP)); % no of tied classes if any
        if tieClasses > 1
            for ti=1:cm
                if classStructure(ti,2)==maxP
                    predictedC=classStructure(ti,1);
                    break;
                end
            end
            accuracy=1/tieClasses;
        else
            if predictedC == actualClass
                accuracy = 1.0; 
            end
        end
        prob=maxP/evidence;
        if evidence==0
            prob=0;
        end
        classification_accuracy=classification_accuracy+accuracy;
        fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',i-1, predictedC, prob, actualClass, accuracy);
    end
    fprintf('classification accuracy=%6.4f\n', classification_accuracy/tm); 
end

function histograms(tline,testFile,classes,number,classesLength)
    [m,n] = size(tline);
    [cm,cn] = size(classes);
    bin = str2num(number);
    S=Inf;
    L=-Inf;
    LargeSmallG = zeros(cm*(n-1)*bin,11);
    rowStart=1;
    unset=0;
    for c=1:cm
        for i=1:n-1
            for j=1:m
                if classes(c,1) == tline(j,n)
                    % calculating the S and L for every class and storing
                    % it in the LargeSmallG matrix for later use
                    if unset==0
                        S = tline(j,i);
                        L = tline(j,i);
                        unset=1;
                    elseif tline(j,i) < S
                        S=tline(j,i);
                    elseif tline(j,i) > L
                        L=tline(j,i);
                    end
                end
            end
            for b=1:bin
                LargeSmallG(rowStart,1)=classes(c,1); % class
                LargeSmallG(rowStart,2)=i-1; % dimension
                LargeSmallG(rowStart,3)=b-1; % binNumber
                LargeSmallG(rowStart,4)=S; % S
                LargeSmallG(rowStart,5)=L; % L
                LargeSmallG(rowStart,9)=Inf; % width by default for bins 0 and n-1
                G=(L-S)/(bin-3);
                if G < 0.0001
                    G=0.0001;
                end
                LargeSmallG(rowStart,6)=G; % G
                if b == 1
                    LargeSmallG(rowStart,7)=-Inf; % LowerRange for Bin 0
                    LargeSmallG(rowStart,8)=S-G/2; % UpperRange for Bin 0
                    
                elseif b == bin
                    LargeSmallG(rowStart,7)=LargeSmallG(rowStart-1,8); % LowerRange for Bin-1
                    LargeSmallG(rowStart,8)=Inf; % UpperRange for Bin-1
                else
                    LargeSmallG(rowStart,7)=LargeSmallG(rowStart-1,8); % LowerRange any other Bin
                    LargeSmallG(rowStart,8)=S+(b-2)*G+(G/2); % UpperRange for any other Bin
                    LargeSmallG(rowStart,9)=G;
                end
                rowStart=rowStart+1;
            end
            unset=0;
        end
    end
    BinMatrixCount = zeros(bin,n-1);
    
    % Iterating over original data set to obtain count of how many
    % dimensions in each class belong to each bin
    for i=1:n-1
        for j=1:m
            class = tline(j,n);
            if class > cm
                %special case to handle satellite file having no class 6
                %between 5 and 7
                class = class-1;
            end
            startClass = class*bin*(n-1)-bin*(n-1)+1;
            endD = startClass + (bin*i);
            startD = endD - bin;
            for b=startD:endD-1
                if tline(j,i) >= LargeSmallG(b,7) && tline(j,i) < LargeSmallG(b,8)
                    LargeSmallG(b,10)=LargeSmallG(b,10)+1;
                    BinMatrixCount(LargeSmallG(b,3)+1,i) = BinMatrixCount(LargeSmallG(b,3)+1,i)+1;
                end
            end
        end
    end
	%disp(LargeSmallG);
    
    %Calculating final P(bin|class) for each class,attribute,bin using
    %bayes formula where P(bin|class) =P(class|bin)*P(bin)/P(class)
    finalLoopEnd = cm*(n-1)*bin;
    for i=1:finalLoopEnd
        class=LargeSmallG(i,1);
        if class > cm
                %special case to handle satellite file having no class 6
                %between 5 and 7
                class = class-1;
        end
        binNumber=LargeSmallG(i,3)+1;
        attribute=LargeSmallG(i,2)+1;
        Pbin=BinMatrixCount(binNumber,attribute)/(m*LargeSmallG(i,9));
        if LargeSmallG(i,9) == Inf
            Pbin=0;
        end
        PClassBin=LargeSmallG(i,10)/BinMatrixCount(binNumber,attribute);
        if BinMatrixCount(binNumber,attribute) == 0
            PClassBin=0;
        end
        PClass = classesLength(class,1)/m;
        LargeSmallG(i,11)=PClassBin*Pbin/PClass;
    end
    for i=1:finalLoopEnd
        fprintf('Class %d, attribute %d, bin %d, P(bin | class) = %.2f\n',LargeSmallG(i,1),LargeSmallG(i,2),LargeSmallG(i,3),LargeSmallG(i,11));
    end
    
    %Classification of histograms:
    classification_accuracy=0;
    [tm,tn]=size(testFile);
    for i=1:tm
        classStructure = zeros(cm,2);
        actualClass=testFile(i,tn);
        classStructure(:,1)=classes(:,1); %storing all classes with their class numbers in this structure
        evidence=0;
        for c=1:cm
            posteriorProb=classesLength(c)/m;
            for j=1:tn-1
                dimension=j-1;
                element=testFile(i,j);
                for mainLoop=1:finalLoopEnd
                    if LargeSmallG(mainLoop,1)==classes(c) && dimension==LargeSmallG(mainLoop,2) && element>=LargeSmallG(mainLoop,7) && element<LargeSmallG(mainLoop,8)
                        posteriorProb=posteriorProb*LargeSmallG(mainLoop,11);
                    end
                end
            end
            evidence=evidence+posteriorProb;
            classStructure(c,2)=posteriorProb;
        end
        accuracy=0.0;
        maxP=max(classStructure(:,2));
        predictedC=classStructure(classStructure(:,2)==maxP,1);
        tieClasses = length(find(classStructure(:,2)==maxP)); % no of tied classes if any
        if tieClasses > 1
            for ti=1:cm
                if classStructure(ti,2)==maxP
                    predictedC=classStructure(ti,1);
                    break;
                end
            end
            accuracy=1/tieClasses;
        else
            if predictedC == actualClass
                accuracy = 1.0; 
            end
        end
        classification_accuracy=classification_accuracy+accuracy;
        prob=maxP/evidence;
        if evidence == 0
            prob=0;
        end
        fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',i-1, predictedC, prob, actualClass, accuracy);
    end
    fprintf('classification accuracy=%6.4f\n', classification_accuracy/tm);   
end

function gaussians(tline,testFile,classes,classesLength)
    [m,n] = size(tline);
    [cm,cn] = size(classes);
    Sum = zeros(cm,n-1);
    Count = zeros(cm,n-1);
    Mean = zeros(cm,n-1);
    %Iterate over data and get the mean and standard deviation
    for i=1:m
        for j=1:n-1
            class=find(classes==tline(i,n));
            Sum(class,j)= Sum(class,j) + tline(i,j);
            Count(class,j)= Count(class,j) + 1;
            Mean(class,j)= Sum(class,j)/Count(class,j);
        end
    end
    Variance = zeros(cm,n-1);
    %calculating standard deviation using the variance formula
    for i=1:m
        for j=1:n-1
            class=find(classes==tline(i,n));
            value = tline(i,j) - Mean(class,j);
            Variance(class,j) = Variance(class,j) + value*value/(Count(class,j)-1);
        end
    end
    for i=1:cm
        for j=1:n-1
            if Variance(i,j) < 0.0001
                Variance(i,j) = 0.0001;
            end
        end
    end
    Variance = Variance.^(1/2);
    for i=1:cm
        for j=1:n-1
            fprintf('Class %d, attribute %d, mean = %.2f, std = %.2f\n',classes(i), j-1, Mean(i,j), Variance(i,j));
        end
    end
    
    %gaussain classification
    classification_accuracy=0;
    newPI= 3.1415926535897932384626;
    [tm,tn]=size(testFile);
    for i=1:tm
        classStructure = zeros(cm,2);
        actualClass=testFile(i,tn);
        classStructure(:,1)=classes(:,1); %storing all classes with their class numbers in this structure
        evidence=0;
        for c=1:cm
            posteriorProb=classesLength(c)/m;
            for j=1:n-1 % going over each dimension in this row
                mean=Mean(c,j);
                std=Variance(c,j); %actually std just called Variance
                expFactor = exp(-1*(testFile(i,j)-mean)*(testFile(i,j)-mean)/(2*std*std));
                posteriorProb = posteriorProb*expFactor/((2*newPI*std*std).^(1/2));
            end
            evidence=evidence+posteriorProb;
            classStructure(c,2)=posteriorProb;
        end
        accuracy=0.0;
        maxP=max(classStructure(:,2));
        predictedC=classStructure(find(classStructure(:,2)==maxP),1);
        tieClasses = length(find(classStructure(:,2)==maxP)); % no of tied classes if any
        if tieClasses > 1
             for ti=1:cm
                if classStructure(ti,2)==maxP
                    predictedC=classStructure(ti,1);
                    break;
                end
            end
            accuracy=1/tieClasses;
        else
            if predictedC == actualClass
                accuracy = 1.0;
            end
        end
        classification_accuracy=classification_accuracy+accuracy;
        fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',i-1, predictedC, maxP/evidence, actualClass, accuracy);
    end
    fprintf('classification accuracy=%6.4f\n', classification_accuracy/tm);
end