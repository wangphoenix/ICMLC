clear all;

load G:\毕业实验成果\数据\1.GCRMA处理后的数据\2.Prostate\GCRMA_t_no_rowcol.txt a
load G:\毕业实验成果\数据\1.GCRMA处理后的数据\2.Prostate\class.txt
%用L1/2的算法,lqfit2是优化过的
%[result] = lqfit2(GCRMA_t_class_no_rowcol,class)

options.verbose=1;
options.lambda=0.001 ;% regul parameter
options.theta=.01; % parameter for lsp
options.p=.5; % parameter for lp
options.reg='lp'; % l2 l1 lp, lsp are possible options
lambdas=0.0001:0.0002:0.01;%colon式子的lam
%lam=0.0029有12610个0，也就是有15个被选出的特征，但是对于此数据集，很多探针重复，都选出重复的
%因此再取lam=0.0027，有12597个0，有28个基因
for i=1:size(lambdas,2)
    options.lambda=lambdas(i)
    [svmlp]=gist_hinge2(GCRMA_t_no_rowcol,class,options);
     U(i,:)=(svmlp.w(:,1))'
end
%sum(U==0,2) 获取每次结果0的数量
%I=find(U(10,:)~=0) 获取第10行不为0的所有下标