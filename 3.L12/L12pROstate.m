clear all;

load prostate.txt
load prostate_result2.txt
%��L1/2���㷨,lqfit2���Ż�����
%[result] = lqfit2(prostate,prostate_result2)

[row,col]=size(prostate)
Init_x = randn(col,1);
MaxIterNum = 2000;
lambda =0.02;%����ֻȡһ��lambda�����Լ���ܿ����
%lambda=0.1:2:10;
[IJT_LHalf_x, IJT_LHalf_cputime] = IJT_LHalf(prostate,prostate_result2,lambda,Init_x,MaxIterNum);
