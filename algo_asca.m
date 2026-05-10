function [Z,A,G,C,F,obj] = algo_asca(X, para)
%min sum \alpha^2 ||X^(v)-W^(v)AZ||_F^2 +lambda1*|| A - GC ||_F^2+lambda2*||AZ-GF||_F^2
%s.t. W^(v)TW^(v)=I,1TZ=1,Z>=0,G^TG=I,C\in Ind,F>=0,F1T=1

%%参数设置
c = para.c;%簇数  
n =para.n;%样本数
V= para.V;%视图数
d =para.d;% 降维维度
m = para.m; % 锚点数
lambda1 = para.lambda1; % 锚点聚类
lambda2 = para.lambda2; % 样本聚类
max_iter=para.max_iter; % 最大迭代次数
%conv_threshold = para.conv_threshold;%收敛阈值


%% 初始化变量

% 初始化 W^(v)
W = cell(V, 1);
for v = 1:V
     di = size(X{v},1);         
     W{v} = zeros(di, d);   
end

A=eye(d,m);

Z=zeros(m,n);
Z(:,1:m) = eye(m);
F=zeros(c,n);

G=eye(d,c);
%初始化C
C=zeros(c,m);
j=1;
for i=1:c
    C(i,j:j+m/c-1)=1;
    j=j+m/c;
end

alpha = ones(1,V);%/V; % 实际不加权
AZ=A*Z;


% 主迭代
for iter = 1:max_iter
    %% 更新 W^(v)
    W_temp=AZ';
    for v = 1:V
        [U_W, ~, V_W] = svd(X{v}*W_temp, 'econ');
        W{v} = U_W * V_W';
    end

    sumAlpha = 0; 
    A_temp = 0;     
    for iv = 1:V
        al2 = alpha(iv)^2;    
        sumAlpha = sumAlpha + al2;
        A_temp = A_temp + al2 * W{iv}' * X{iv} * Z';
    end

    %% 2. 更新 A
    GC=G*C;
    A_P=(sumAlpha+lambda2)*(Z*Z')+lambda1*eye(m)+1e-8*eye(m);
    A_Q=A_temp+lambda1*GC+lambda2*G*F*Z';
    A=A_Q/A_P;

    %% 3.更新G
    [U_G, ~, V_G] = svd(lambda1*A*C'+lambda2*A*Z*F', 'econ');
    G = U_G * V_G';


    %% 4. 更新 C
    C=zeros(c,m);
    for im=1:m
        diffs = A(:,im)' - G';     
        dists = sum(diffs.^2,2);  
        [~, idx] = min(dists);    
        colC = zeros(c,1);        
        colC(idx) = 1;
        C(:,im) = colC;           
    end


    %% 5.更新 Z
    Q_Z = (sumAlpha+lambda2)* (A'*A)+1e-8*eye(m);
    P_Z=0;
    for v=1:V
      P_Z = P_Z+alpha(v)^2*A'*W{v}'*X{v};
    end
    P_Z=P_Z+lambda2*A'*G*F;
    H_quad = 2 * Q_Z;       
    %ADMM参数
    rho=1;
    InvMat = inv(H_quad + rho * eye(m));   
    parfor i = 1:n
        f_quad = -2 * P_Z(:, i);  
        Z(:, i) = admm_qp_simplex(InvMat, f_quad, rho, 50);
    end

    AZ=A*Z;
    %% 6.更新 F
    F=G'*AZ;
    for i = 1:n 
        F(:, i) = EProjSimplex_new(F(:, i), 1);
    end
    

    % %% 6.更新权重（不更新）
    % 
    % alpha_temp = zeros(V,1);  
    % parfor iv = 1:V
    %     alpha_temp(iv) = norm(X{iv} - W{iv}*A*Z, 'fro')^2;  
    % end
    % Mfra = alpha_temp.^-1;        
    % Q = 1 / sum(Mfra);   
    % alpha = Q * Mfra;    

    %% 计算目标函数值
    obj.obj1(iter) = 0;
    %重构误差
    for v = 1:V
        obj.obj1(iter) = obj.obj1(iter) + norm( X{v} - W{v} * AZ, 'fro')^2;
    end
    % 锚点聚类
    obj.obj2(iter)=lambda1 * norm( A - G*C , 'fro')^2;
    %样本聚类
    obj.obj3(iter)=lambda2* norm( AZ - G*F , 'fro')^2;

    obj.all(iter) = obj.obj1(iter)+obj.obj2(iter)+obj.obj3(iter);

    % 收敛判断，实际固定20次
    if iter > 1
        rel_change = obj.all(iter) - obj.all(iter-1);
        fprintf('Iteration %d, Objective Value: %.4f, Relative Change: %.6f\n', iter, obj.all(iter), rel_change);
        % if abs(rel_change) < conv_threshold
        %     fprintf('Converged at iteration %d\n', iter);
        %     break;
        % end
end
    
end