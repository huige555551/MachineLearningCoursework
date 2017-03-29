function dtree(training_file, test_file, option, pruning_thr)
    tFile = importdata(training_file);
    testFile = importdata(test_file);
    pruning=str2num(pruning_thr);
    switch(option)
        case 'optimized'
            generateDtree(tFile,testFile,'optimized',pruning);
        case 'randomized'
            generateDtree(tFile,testFile,'randomized',pruning);
        case 'forest3'
            generateDForest(tFile,testFile,3,'randomized',pruning);
        case 'forest15'
            generateDForest(tFile,testFile,15,'randomized',pruning);
        otherwise
            disp('Invalid option parameter!');
    end
end

function infoMatrix=recursivelyDisplayTreeInfo(treeNumber,obj,infoMatrix)
    infoMatrix.info = [ infoMatrix.info ; [treeNumber obj.node obj.attribute obj.threshold obj.gain]];
    if obj.attribute ~= -1
        infoMatrix=recursivelyDisplayTreeInfo(treeNumber,obj.left_child,infoMatrix);
        infoMatrix=recursivelyDisplayTreeInfo(treeNumber,obj.right_child,infoMatrix);
    end
end

function generateDForest(tFile,testFile,numberOfTrees,option,pruning)
    [~,n]=size(tFile);
    default=mode(tFile(:,n));
    uniqueClasses = unique(tFile(:,n));
    [cn,~]=size(uniqueClasses);
    forest =struct('tree',[]);
    for i=1:numberOfTrees
        forest.tree = [ forest.tree ; DTL(i-1,uniqueClasses,tFile,default,pruning,option,1)];
    end
    
    for f=1:numberOfTrees
        %printing output in BFS format for each tree in forest
        infoMatrix = recursivelyDisplayTreeInfo(f-1,forest.tree(f),struct('info',[]));
        sObj = sortrows(infoMatrix.info,2);
        [sm,~]=size(sObj);
        for i=1:sm
            fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n',sObj(i,:));
        end
    end
    
    %classification for density forests
    [tm,tn]=size(testFile);
    classification_accuracy=0;
    for i=1:tm
        actualClass=testFile(i,tn);
        probDists=zeros(numberOfTrees,cn);
        for f=1:numberOfTrees
            obj=forest.tree(f);
            accuracy=0;
            classified = 0;
            while classified ~= 1
                if obj.attribute ~= -1
                    if testFile(i,obj.attribute) < obj.threshold
                        obj=obj.left_child;
                    else
                        obj=obj.right_child;
                    end
                else
                    %leaf node reached
                    classified =1;
                    probDists(f,:) = obj.default';
                end
            end
        end
        %main classification logic
        probFinal=mean(probDists);
        prob=max(probFinal);
        predictedClass = uniqueClasses(probFinal==prob);
        ties = length(find(probFinal==prob));
        if ties >1
            [pm,~]=size(predictedClass);
            if sum(find(predictedClass==actualClass))==1
                accuracy = 1/pm;
            end
            predictedClass = predictedClass(1,1);
        else  
            if predictedClass == actualClass
                accuracy=1;
                classification_accuracy=classification_accuracy+1;
            end
        end
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n',i-1,predictedClass,actualClass,accuracy);
    end
    fprintf('classification accuracy=%6.4f\n', classification_accuracy/tm);
end

function generateDtree(tFile,testFile,option,pruning)
    [~,n]=size(tFile);
    default=mode(tFile(:,n));
    uniqueClasses = unique(tFile(:,n));
    finalTree = DTL(0,uniqueClasses,tFile,default,pruning,option,1);
    
    %printing output in BFS format
    infoMatrix = recursivelyDisplayTreeInfo(0,finalTree,struct('info',[]));
    sObj = sortrows(infoMatrix.info,2);
    [sm,~]=size(sObj);
    for i=1:sm
        fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n',sObj(i,:));
    end
    
    %classification for density tree
    [tm,tn]=size(testFile);
    classification_accuracy=0;
    for i=1:tm
        classified = 0;
        actualClass =testFile(i,tn);
        obj=finalTree;
        prob=0;
        predictedClass =-1;
        accuracy=0;
        while classified ~= 1
            if obj.attribute ~= -1
                if testFile(i,obj.attribute) < obj.threshold
                    obj=obj.left_child;
                else
                    obj=obj.right_child;
                end
            else
                %leaf node reached
                classified =1;
                prob=max(obj.default);
                predictedClass = uniqueClasses(obj.default==prob);
                ties = length(find(obj.default==prob));
                if ties >1
                    [pm,~]=size(predictedClass);
                    if sum(find(predictedClass==actualClass))==1
                        accuracy = 1/pm;
                    end
                    predictedClass = predictedClass(1,1);
                else  
                    if predictedClass == actualClass
                        accuracy=1;
                        classification_accuracy=classification_accuracy+1;
                    end
                end
            end
        end
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n',i-1,predictedClass,actualClass,accuracy);
    end
    fprintf('classification accuracy=%6.4f\n', classification_accuracy/tm);
end

function tree = DTL(treeNum,unique_classes,examples,default,pruning,option,nodeVal)
    [m,n] = size(examples);
    max_gain = -1;
    best_attribute=-1;
    best_threshold =-1;
    eCn=0;
    if n>0
        uniqueExampleClasses = unique(examples(:,n));
        [eCn,~]=size(uniqueExampleClasses);
    end
    if m < pruning || eCn == 1
        %fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n',treeNum, nodeVal, best_attribute, best_threshold, max_gain);
        tree = struct('attribute',best_attribute,'threshold',best_threshold,'gain',max_gain,'node',nodeVal,'default',default);
    else
        [cn,~]=size(unique_classes);
        classStruct = zeros(cn,1);
        if strcmp(option, 'optimized')
            [max_gain,best_attribute,best_threshold] = chooseAttributeOptimized(examples,m);
        else
            [max_gain,best_attribute,best_threshold] = chooseAttributeRandomized(examples,m);
        end
        tree = struct('attribute',best_attribute,'threshold',best_threshold,'gain',max_gain,'node',nodeVal);
        left=struct('examples',[],'classes',classStruct);
        right=struct('examples',[],'classes',classStruct);
        for i=1:m
            classIndex = find(unique_classes == examples(i,n));
            if examples(i,best_attribute) < best_threshold
                left.examples=[left.examples; examples(i,:)];
                left.classes(classIndex)=left.classes(classIndex)+1;
            else
                right.examples=[right.examples; examples(i,:)];
                right.classes(classIndex)=right.classes(classIndex)+1;
            end
        end
        %Calculating distribution for classes in left and right child nodes
        [lm,~]=size(left.examples);
        [rm,~]=size(right.examples);

        lSum=sum(left.classes(:,1));
        rSum=sum(right.classes(:,1));

        if  lm ~= 0 && lm > pruning
            left.classes(:,1)=left.classes(:,1)/lSum;
        else
            left.classes =default;
        end
        if rm ~= 0 && rm > pruning
            right.classes(:,1)=right.classes(:,1)/rSum;
        else
            right.classes =default;
        end
        %making the recursive calls to DTL function for left and right
        %child nodes
        %fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n',treeNum, nodeVal, best_attribute-1, best_threshold, max_gain);
        tree.left_child = DTL(treeNum,unique_classes,left.examples,left.classes,pruning,option,nodeVal*2);
        tree.right_child = DTL(treeNum,unique_classes,right.examples,right.classes,pruning,option,((nodeVal*2)+1));
    end
end

%function for choosing attribute in optimized version
function [max_gain,best_attribute,best_threshold] = chooseAttributeOptimized(examples,parentCount)
    max_gain = -1;
    best_attribute = -1;
    best_threshold = -1;
    [~,n] = size(examples);
    for i=1:n-1
        attribute_values = examples(:,i);
        L = min(attribute_values);
        M = max(attribute_values);
        for k = 1:50
            threshold = L + k*(M-L)/51;
            gain = calculateInformationGain(examples,i,threshold,parentCount);
            if gain > max_gain
                max_gain = gain;
                best_attribute = i;
                best_threshold = threshold;
            end
        end
    end
end

%function for choosing attribute in randomized version
function [max_gain,best_attribute,best_threshold] = chooseAttributeRandomized(examples,parentCount)
    max_gain = -1;
    best_attribute = -1;
    best_threshold = -1;
    [~,n] = size(examples);
    %randomly selecting an attribute between 1 to n-1
    i=randi([1 n-1],1,1);
    attribute_values = examples(:,i);
    L = min(attribute_values);
    M = max(attribute_values);
    for k = 1:50
        threshold = L + k*(M-L)/51;
        gain = calculateInformationGain(examples,i,threshold,parentCount);
        if gain > max_gain
            max_gain = gain;
            best_attribute = i;
            best_threshold = threshold;
        end
    end
end

%parse through examples and calculate Information Gain
function gain = calculateInformationGain(examples,n,threshold,parentCount)
    [totalCount,en] =size(examples);
    
    unique_classes = unique(examples(:,en));
    [cn,~] = size(unique_classes);
    
    classStruct = zeros(cn,1);
    total =struct('classes',classStruct,'count',totalCount);
    left = struct('classes',classStruct,'count',0);
    right = struct('classes',classStruct,'count',0);
    for i=1:totalCount
        classIndex = find(unique_classes == examples(i,en));
        total.classes(classIndex)=total.classes(classIndex)+1;
        if examples(i,n) < threshold
            left.count =left.count+1;
            left.classes(classIndex)=left.classes(classIndex)+1;
        else
            right.count =right.count+1;
            right.classes(classIndex)=right.classes(classIndex)+1;
        end
    end
    %calculating final gain value
    gain = getEntropy(total,parentCount) - (left.count/parentCount)*getEntropy(left,left.count) - (right.count/parentCount)*getEntropy(right,right.count);
end

%calcuate entropy will use struct to 
function entropy = getEntropy(obj,count)
    entropy=0;
    [cn,~]=size(obj.classes);
    for i=1:cn
        if obj.classes(i) > 0
            entropy= entropy + ((-obj.classes(i)/count) * (log2(obj.classes(i)/count)));
        end
    end     
end