function z = admm_qp_simplex(InvMat, p, rho, max_iter)
% 使用 ADMM 求解单纯形约束二次规划
% 求解: min 0.5*z'*Q*z + p'*z  s.t. z >= 0, sum(z)=1


    if nargin < 3, rho = 1.0; end
    if nargin < 4, max_iter = 20; end

    m = length(p);
  
    %InvMat = inv(Q + rho * eye(m));
    
    %初始化
    z = ones(m, 1) / m;
    u = z;
    y = zeros(m, 1);
    
    %迭代
    for k = 1:max_iter
        % 更新z
        rhs = rho * u - y - p;
        z = InvMat * rhs;
        
        %更新u
        v = z + y / rho;
        u = EProjSimplex_new(v, 1);
        
        % 更新y
        y = y + rho * (z - u);
        
        % 收敛检测
         if norm(z - u) < 1e-3, break; end
    end
    
    % 返回u
    z = u;
end