clear all;

load G:\��ҵʵ��ɹ�\����\1.GCRMA����������\4.GLI-85\GCRMA_t_no_rowcol.txt
load G:\��ҵʵ��ɹ�\����\1.GCRMA����������\4.GLI-85\class.txt
%��L1/2���㷨,lqfit2���Ż�����
%[result] = lqfit2(GCRMA_t_class_no_rowcol,class)

options.verbose=1;
options.lambda=0.001 ;% regul parameter
options.theta=.01; % parameter for lsp
options.p=.5; % parameter for lp
options.reg='lp'; % l2 l1 lp, lsp are possible options
%03048
lambdas=03048;%colonʽ�ӵ�lam
%lambdas=0.0001:0.0002:0.01;%colonʽ�ӵ�lam
%lam=0.0029��12610��0��Ҳ������15����ѡ�������������Ƕ��ڴ����ݼ����ܶ�̽���ظ�����ѡ���ظ���
%�����ȡlam=0.0027����12597��0����28������
for i=1:size(lambdas,2)
    options.lambda=lambdas(i)
    [svmlp]=gist_hinge2(GCRMA_t_no_rowcol,class,options);
     U(i,:)=(svmlp.w(:,1))'
end
%sum(U==0,2) ��ȡÿ�ν��0������
%I=find(U(10,:)~=0) ��ȡ��10�в�Ϊ0�������±�