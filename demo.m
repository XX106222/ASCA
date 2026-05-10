clear all;
%clc;
addpath(genpath('./'));

dsPath='D:/对比方法/dataset_XY/';
ds = {
'MSRC',...
'BBCSport',...
'Wiki_fea',...
'CCV',...
'BBC',...
'BDGP_fea',...
'caltech101_8677',...
'AWAfea',...%8
'MNIST_fea',...
'Caltech101-20',...%10
'Notting-Hill',...
'Hdigit1',...
'ALOI',...%13
'Animal',...
'YoutubeFace_sel',...%15
'synthetic3d',...
'ORL',...
'NGs',...
'WebKB',...
'HW'...%20
'TwoMoon',...
'ThreeRing',...
'3sources',...
'100leaves',...
'NUSWIDEOBJ',...%25
'Caltech256_fea',...
'cifar10',...
'cifar100',...
'NoisyMNIST_MVC',...
'ImageNet100',...%30
'yale',...
'ACM',...
'CiteSeer',...
'UCI_Digits'
};
respath='res/';
if ~exist(respath, 'dir')
mkdir(respath);
end
respath_max='res/res_max/';
if ~exist(respath_max, 'dir')
mkdir(respath_max);
end
logFileName = fullfile(respath, 'runninglog.txt');

para.lambda1 = 0; 
para.lambda2 = 0;
para.max_iter = 20; % 最大迭代次数
%para.conv_threshold = 1e-80000;

%默认BN，从'no','unit','BN'选
para_perp={'unit','no','unit','BN','no',...%5
            'unit','unit','BN','no','BN',...%10
           'unit','unit','BN','unit','BN',...%15
           'no','unit','unit','no','BN',...%20
           'BN','BN','unit','unit','BN',...%25
           'BN','no','unit','no','no',...%30
           'BN','BN','BN','BN','unit'};
for di=1
    dataName = ds{di}; 
    fid = fopen(logFileName, 'a+');
    fprintf(fid, '==================== Dataset: %s ====================\n', dataName);
    fclose(fid);
    disp(dataName);
    load(strcat(dsPath,dataName));
    true_label =Y'; % 真实标签
    para.V = length(X); %视图数
    for v = 1:para.V
        X{v} = X{v}';
    end
    para.c = length(unique(true_label)); %类数
    para.n = size(X{1}, 2); %样本数

    perp_f=para_perp{di};
    [X] = data_prep(X,perp_f);%预处理:'BN' 'LN' 'MM1' 'MM2' 'unit' 'no'
    
    
    para_l1=[0.001,0.01,0.1,1,10,100,1000];
    para_l2=[0.001,0.01,1,100,1000];
    maxACC=0;
   
    for i_m=1:5
        for i_i=1:7
            for i_j=1:5
              para.d=para.c*i_m;
              para.m=para.c*i_m;
              para.lambda1 =para_l1(i_i); 
              para.lambda2 =para_l2(i_j); 
              tic;
              [Z,A,G,C,F,obj] = algo_asca(X, para);
              [U, singa, ~] = svd(Z', 'econ');
              res = myNMIACCwithmean(U,true_label,para.c);

              acc(i_m,i_i,i_j)=res(1);
              nmi(i_m,i_i,i_j)=res(2);
              purity(i_m,i_i,i_j)=res(3);
              fscore(i_m,i_i,i_j)=res(4);
              time(i_m,i_i,i_j)=toc;

              outStr = sprintf('acc:%.4f,nmi:%.4f,purity:%.4f,fscore:%.4f,time:%.4f,m:%d,i:%d,j:%d\n', ...
                    acc(i_m,i_i,i_j),nmi(i_m,i_i,i_j),purity(i_m,i_i,i_j), ...
                    fscore(i_m,i_i,i_j),time(i_m,i_i,i_j),i_m,i_i,i_j);
              disp(outStr); % 屏幕显示
              fid = fopen(logFileName, 'a+');
              fprintf(fid, '%s\n', outStr);
              fclose(fid);
                
              dataset_name=[respath,dataName,'-',perp_f,'.mat'];
              save(dataset_name,"acc","nmi","purity","fscore","time");

              currentACC = res(1);
              if currentACC > maxACC
                maxACC = currentACC;
                dataset_name=[dataName,'-',perp_f,'.mat'];
                save([respath_max,dataset_name],'Z','A',"G","C",'time','res','obj','F','para','Y');
              end 

            end
        end
    end

    [maxVal, linearIdx] = max(acc, [], 'all'); 
    [best_m,best_i, best_j] = ind2sub(size(acc), linearIdx); 
    fprintf('max_acc:%.4f,best_m:%d,best_lambda:%.4f,best_gamma:%.4f\n' ...
        ,maxVal,best_m*para.c,para_l1(best_i),para_l2(best_j));

    clear acc nmi purity fscore time;
end
