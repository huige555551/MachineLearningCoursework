function linear_regression(trainingData,degree,lambda)
 tData=importdata(trainingData);
 [tm,tn]=size(tData);
 M=str2num(degree)+1;
 l=str2num(lambda);
 iMatrix=eye(M);
 phiMatrix=ones(tm,M);
 for i=2:M
     for j=1:tm
         phiMatrix(j,i)=tData(j,1)^(i-1);
     end
 end
 t=tData(:,tn);
 phiMatrixTranspose=transpose(phiMatrix);
 w=inv((l*iMatrix) + (phiMatrixTranspose*phiMatrix))*(phiMatrixTranspose*t);
 [wm,wn]=size(w);
 for i=1:wm
     fprintf('w%d=%.4f\n',(i-1),w(i,1));
 end
 if wm < 3
     fprintf('w2=0\n');
 end