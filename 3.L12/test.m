%@ Edited by Jinshan Zeng, School of Mathematics&Stastics, Xi'an Jiaotong University, 
% email: jsh.zeng@gmail.com, date: Sep. 27, 2014
%Test code for IJT algorithm
% Applying IJT algorithm to the sparse signal recovery problem

%其实里面的方法描述挺好的，但是就是要调成logistic回归，要看论文公式去调


clear all; close all; clc;
%
k=15; % Sparsity Level
n = 500; % 500特征数dimension of signal
m = 256; % 256样本数number of measurements

A = randn(m,n)/sqrt(m); % 相当于数据集，256样本X500个特征Generate the gaussian matrix

% Generate the sparse signal
x0 = zeros(n,1);%每个特征的权重，500个特征的权重
Tr_C = 2*randn(k,1);
RandP = randperm(n);
x0(RandP(1:k)) = Tr_C;

y = A*x0; %256样本对应的值 Generate the measurements
%lambda =0.015;%这里只取一个lambda，所以计算很快完成
lambda =0.2%调大一点也可以很稀疏
%Init_x = zeros(n,1);
Init_x = randn(n,1);
MaxIterNum = 2000;
% Recovered by IJT algorithm
[IJT_LHalf_x, IJT_LHalf_cputime] = IJT_LHalf(A,y,lambda,Init_x,MaxIterNum);
[IJT_L2rds_x, IJT_L2rds_cputime] = IJT_L2rds(A,y,lambda,Init_x,MaxIterNum);

IJT_LHalf_RecErr = norm(IJT_LHalf_x -x0)^2; % Recovery error of IJT algorithm for L1/2 regularization
IJT_L2rds_RecErr = norm(IJT_L2rds_x -x0)^2; % Recovery error of IJT algorithm for L2/3 regularization

figure, plot(IJT_LHalf_x,'r-*');
hold on; plot(IJT_L2rds_x,'b-o');
hold on; plot(x0, 'k-d');
legend('IJT for L1/2', 'IJT for L2/3', 'True Signal');
axis tight;
