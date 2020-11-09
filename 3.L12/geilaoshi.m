clear all;

data = load("G:\毕业实验成果\给老师的\DLBCL.txt")
class=data(:,11)
sample=data(:,1:10)
%sample=sample'
%data = load("G:\毕业实验成果\给老师的\DLBCL.txt")
%用L1/2的算法,lqfit2是优化过的
%[result] = lqfit2(GCRMA_t_class_no_rowcol,class)

options.verbose=1;
options.lambda=0.001 ;% regul parameter
options.theta=.01; % parameter for lsp
options.p=.5; % parameter for lp
options.reg='lp'; % l2 l1 lp, lsp are possible options
lambdas=0.08:0.005:0.12
for i=1:size(lambdas,2)
    options.lambda=lambdas(i)
    [svmlp]=gist_hinge2(sample,class,options);
     U(i,:)=(svmlp.w(:,1))'
end
%sum(U==0,2) 获取每次结果0的数量
%I=find(U(10,:)~=0) 获取第10行不为0的所有下标