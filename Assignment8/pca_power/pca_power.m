function pca_power(training_file, test_file, M, iter)
    tFile = importdata(training_file);
    testFile = importdata(test_file);
    M=str2num(M);
    iterations=str2num(iter);
    [m,n]=size(tFile);
    A=tFile(:,1:end-1);
    ud=ones(n-1,M);
    % main loop runs for the number of iterations specified in the command
    % window
    for d=1:M
        Sd=cov(A); %Covariance matrix for this step
        for i=1:iterations
            %power method for obtaining ud
            ud(:,d) = (Sd*ud(:,d))/calculateDist(Sd*ud(:,d));
        end
        for r=1:m
            A(r,:)=A(r,:) - ud(:,d)'*A(r,:)'*ud(:,d)';
        end
    end
    
    %displaying the training results
    for d=1:M
        fprintf('Eigenvector %d\n',d);
        [um,~]=size(ud(:,d));
        for i=1:um
            fprintf('%3d: %.4f\n',i,ud(i,d));
        end
    end
    
    % projection on test data
    [tm,~]=size(testFile);
    projMatrix = ud';
    for i=1:tm
        fprintf('Test object %d\n',i-1);
        xn=testFile(i,1:end-1);
        for d=1:M
            fprintf('%3d: %.4f\n',d,projMatrix(d,:)*xn');
        end
    end
end

function dist =  calculateDist(vec)
    [vm,~]=size(vec);
    dist=0.0;
    for i=1:vm
        dist = dist + vec(i,1)*vec(i,1);
    end
    dist = dist.^(1/2);
end