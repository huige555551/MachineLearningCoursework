function svd_power(data_file, M, iter)
    A = importdata(data_file);
    M=str2num(M);
    iterations=str2num(iter);
    [m,n]=size(A);
    AAt=A*A';
    AtA=A'*A;
    [uM,~]=size(AAt);
    [vM,~]=size(AtA);
    U=powerMethod(AAt,uM,M,iterations);
    V=powerMethod(AtA,vM,M,iterations);
    S=zeros(M,M);
    
    
    
    for i=1:M
        temp =AAt*U(:,i);
        maxVal = max(temp);
        position = temp==maxVal;
        S(i,i) = (maxVal/U(position,i)).^(1/2);
    end
    AFinal = U*S*V';
    
    displayResult('Matrix U:',U,uM,M);
    displayResult('Matrix S:',S,M,M);
    displayResult('Matrix V:',V,vM,M);
    displayResult('Reconstruction (U*S*V''):',AFinal,m,n);
end

function displayResult(matrixName,matrix,size,M)
    fprintf('\n');
    disp(matrixName);
    for i=1:size
        fprintf('Row %3d: ',i)
        for j=1:M
            fprintf('%8.4f ',matrix(i,j));
        end
        fprintf('\n');
    end
end

function ud = powerMethod(A,n,M,iterations)
    % main loop runs for the number of iterations specified in the command
    % window
    ud = zeros(n,M);
    for i=1:n
        %ud(i,:)=rand(M,1);
        ud(i,:)=1;
    end
    for d=1:M
        Sd=A;
        for i=1:iterations
            %power method for obtaining ud
            ud(:,d) = (Sd*ud(:,d))/calculateDist(Sd*ud(:,d));
        end
        for r=1:n
            A(r,:)=A(r,:) - ud(:,d)'*A(r,:)'*ud(:,d)';
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