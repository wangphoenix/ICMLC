clear all;

load prostate.txt
load prostate_result2.txt
%用L1/2的算法,lqfit2是优化过的
%[result] = lqfit2(prostate,prostate_result2)

[row,col]=size(prostate)
Init_x = randn(col,1);
MaxIterNum = 2000;
lambda =0.02;%这里只取一个lambda，所以计算很快完成
%lambda=0.1:2:10;
[IJT_LHalf_x, IJT_LHalf_cputime] = IJT_LHalf(prostate,prostate_result2,lambda,Init_x,MaxIterNum);
