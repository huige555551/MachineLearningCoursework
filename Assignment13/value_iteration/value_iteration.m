function value_iteration(environment_file,non_terminal_reward,gamma,K)
    fileID = fopen(environment_file);
    k=0;
    structure = string.empty;
    while ~feof(fileID)
        line = strsplit(fgetl(fileID),',');
        structure = [structure; line];
        k=k+1;
    end
    [m,n] = size(structure);
    U = zeros(m,n);
    
    reward = str2double(non_terminal_reward);
    gamma = str2double(gamma);
    itr = str2double(K);
    
    for i=1:m
        for j=1:n
            if strcmp(structure(i,j),'.')
                U(i,j) = reward;
            elseif strcmp(structure(i,j),'X')
                U(i,j) = 0;
            else
                
                U(i,j) = str2double(structure(i,j));
            end
        end
    end
    U1 = U;
    for i=1:itr
        for r=1:m
            for c=1:n
                if U1(r,c) ~= 0 && structure(r,c) == '.'
                    U1(r,c) = reward + gamma * findMaxAction(U,r,c);
                end
            end
        end
        U = U1;
    end
    
    for i=1:m
        for j=1:n-1
            fprintf('%6.3f,',U(i,j));
        end
        fprintf('%6.3f\n',U(i,n));
    end
    fclose(fileID);
end

function maxAction = findMaxAction(U,r,c)
    % initializing boundaries
    [m,n] = size(U);
    
    [ur,uc] = getState(U,r,c,'up',m,n);
    [lr,lc] = getState(U,r,c,'left',m,n);
    [rr,rc] = getState(U,r,c,'right',m,n);
    [dr,dc] = getState(U,r,c,'down',m,n);
    
    %action up
    maxAction = 0.8 * U(ur,uc) + 0.1 * U(lr,lc) + 0.1 * U(rr,rc); 
    
    %action down
    temp = 0.8 * U(dr,dc) + 0.1 * U(lr,lc) + 0.1 * U(rr,rc); 
    if temp > maxAction
        maxAction = temp;
    end
    
    %action left
    temp = 0.8 * U(lr,lc) + 0.1 * U(ur,uc) + 0.1 * U(dr,dc); 
    if temp > maxAction
        maxAction = temp;
    end
    
    %action right
    temp = 0.8 * U(rr,rc) + 0.1 * U(ur,uc) + 0.1 * U(dr,dc);
    if temp > maxAction
        maxAction = temp;
    end
end

function [newr,newc] = getState(U,r,c,action,m,n)
    newr=r;
    newc=c;
    switch(action)
        case 'up'
            newr = r-1;
        case 'down'
            newr = r+1;
        case 'right'
            newc = c+1;
        case 'left'
            newc = c-1;
        otherwise
            newr=r;
            newc=c;
    end
    
    if newr > m || newr < 1
        newr = r;
        newc = c;
    elseif newc > n || newc < 1
        newr = r;
        newc = c;
    elseif U(newr,newc) == 0
        newr = r;
        newc = c;
    end
end