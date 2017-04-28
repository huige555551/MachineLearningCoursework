function k_means_cluster(dataFile,k,iterations)
    tfile = importdata(dataFile);
    K = str2double(k);
    itr = str2double(iterations);
    tfile = tfile(:,1:end-1);
    [m,n] = size(tfile);
    
    %initialization step
    s = zeros(m,n+1);
    s(:,1:end-1) = tfile;
    for i=1:m
        s(i,end) = randi([1,K],1,1);
    end
    
    [clusters,~] = computeNewClusters(s,K,n);
    ErrorKMeans = getError(s,K,clusters);
    fprintf('After initialization: error = %.4f\n', ErrorKMeans);
    
    %main loop begins
    for i=1:itr
        [clusters,~] = computeNewClusters(s,K,n);
        s = assignToClusters(s,K,clusters);
        ErrorKMeans = getError(s,K,clusters);
        fprintf('After iteration %d: error = %.4f\n',i,ErrorKMeans);
    end
end

function s = assignToClusters(s,K,clusters)
    [m,n] = size(s);
    for i=1:m
        distance = getDistance(s(i,:),clusters(1,:),n-1);
        cluster = 1;
        for j=2:K
            if getDistance(s(i,:),clusters(j,:),n-1) < distance
                distance = getDistance(s(i,1:n-1),clusters(j,:),n-1);
                cluster = j;
            end
        end
        s(i,end) = cluster;
    end
end

function error = getError(obj,K,mean)
    [m,n]=size(obj);
    error = 0.0;
    for i=1:K
        for j=1:m
            if obj(j,end) == i
                error = error + getDistance(obj(j,1:end-1),mean(i,:),n-1);
            end
        end
    end
end


function [newClusters,count] = computeNewClusters(obj,K,n)
    newClusters = zeros(K,n);
    count = zeros(K,1);
    for j=1:K
        for i=1:numel(obj(:,1))
            if obj(i,end) == j
                newClusters(j,:) = newClusters(j,:) + obj(i,1:end-1);
                count(j,1) = count(j,1)+1;
            end
        end
        newClusters(j,:)= newClusters(j,:)/count(j,1);
    end
end

% use L2 eucledian distance to calculate distance between two points
function dist = getDistance(a,b,n)
    dist = 0.0;
    for i=1:n
        dist = dist + (b(1,i) - a(1,i)) * (b(1,i) - a(1,i));
    end
    dist = dist .^(1/2);
end