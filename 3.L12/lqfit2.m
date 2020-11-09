function [result]=lqfit2(X,Y)

result=struct();
result.lambdas=[];
result.beta=[];
result.beta0=[];
%result.lambdas=0.1:0.5:20;
result.lambdas=0.1:2:10;

%
nLambda=size(result.lambdas,2);

%n为样本数目
n=size(X,1);

%p为特征值个数
p=size(X,2);

for t=1:nLambda
    fid=fopen('xx.txt','a');
    fprintf(fid,'第%d次lambuda\t',t);
    fprintf(fid,'\r\n');
    fclose(fid);
    [Mu,W,Z,B,B0,BLast,B0Last]=init(X,Y,n,p)
    m=0;
    while 1
        B0=computeB0(X,B,W,Z,n);
        B =computeB(X,B,B0,W,Z,n,p,result.lambdas(1,t));
        if m>0
            result.beta(t,:)=B;
            result.beta0(t,:)=B0;
            break;
        end
        
        if abs(sum(BLast,2)-sum(B,2))<(10^(-8))
            result.beta(t,:)=B;
            result.beta0(t,:)=B0;
            break;
        end
        [Mu,W,Z,flag]=update(X,Y,n,B,B0);
        if flag==1
            result.beta(t,:)=B;
            result.beta0(t,:)=B0;
            break;
        end
        fid=fopen('xx.txt','a');
    fprintf(fid,'%g\t',B);
    fprintf(fid,'\r\n');
    fclose(fid);
        BLast=B;
        B0Last=B0;
        m=m+1;
    %end while
    end
%end for
end
    
%end function
end



function[Mu,W,Z,flag]=update(X,Y,n,B,B0)
flag=0;
for i=1:n
    Mu(i,1)=(exp(B0+X(i,:)*B'))./(1+exp(B0+X(i,:)*B'));
end
for i=1:n
    W(i,1)=Mu(i,1)*(1-Mu(i,1));
end
for i=1:n
    Z(i,1)=(B0+X(i,:)*B')+(Y(i,:)-Mu(i,1))./W(i,1);
    if isnan(Z(i,1))
        flag=1;
        break;
    end
end
end

function[B]=computeB(X,B,B0,W,Z,n,p,lambda)
for k=1:p
    sum1=0;
    sum2=0;
    for i=1:n
        temp1=X(i,:)*B'-X(i,k)*B(1,k);
        temp2=Z(i,1)-B0-temp1;
        temp3=W(i,1)*temp2*X(i,k);
        sum1=sum1+temp3;
    end
    for i=1:n
        temp4=W(i,1)*X(i,k)*X(i,k);
        sum2=sum2+temp4;
    end
    C=sum1./sum2;
    
    r=lambda./sum2;
    
    if abs(C<(0.75*power(r,2./3)))
        B(1,k)=0;
    else
        temp5=acos((r./8)*power(C./3,-1.5));
        temp6=(2./3)*pi-(2./3)*temp5;
        B(1,k)=(2./3)*C*(1+cos(temp6));
    end
    
end
end

function[B0]=computeB0(X,B,W,Z,n)
sum1=0;
sum2=0;
for i=1:n
    temp1=X(i,:)*B';
    temp2=Z(i,1)-temp1;
    temp3=W(i,1)*temp2;
    sum1=sum1+temp3;
end
sum2=sum(W,1);
B0=sum1./sum2;
end

function[Mu,W,Z,b,b0,bLast,b0Last]=init(X,Y,n,p)
b=zeros(1,p);
b0=0;
bLast=b;
b0Last=b0;

Mu=zeros(n,1);
Z=zeros(n,1);
W=zeros(n,1);

for i=1:n
    mu=(exp(b0+X(i,:)*b'))./(1+exp(b0+X(i,:)*b'));
    Mu(i,1)=mu;
end
for i=1:n
    w=Mu(i,1)*(1-Mu(i,1));
    W(i,1)=w;
end
for i=1:n
    z=X(i,:)*b'+b0+(Y(i,:)-Mu(i,1))./(Mu(i,1)*(1-Mu(i,1)));
    Z(i,1)=z;
end

end
