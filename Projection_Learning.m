clear
clc
close all

load('ADNI_info.mat')
MIL_idx = 1;
for idx = 1 : size(ADNI_info,2)
    if sum(isnan(ADNI_info{idx}.VBM_BL))==0 % &&  sum(isnan(ADNI_info{idx}.ADAS_BL))==0
        % ADNI_VBM_MIL{MIL_idx}.Label     = ADNI_info{idx}.ADAS_BL;
        ADNI_VBM_MIL{MIL_idx}.Base      = ADNI_info{idx}.VBM_BL;
        ADNI_VBM_MIL{MIL_idx}.Instance  = [];
        ADNI_VBM_MIL{MIL_idx}.ADAS_BL   = ADNI_info{idx}.ADAS_BL;
        ADNI_VBM_MIL{MIL_idx}.DX_BL     = ADNI_info{idx}.DX_BL;
        ADNI_VBM_MIL{MIL_idx}.FLU_BL    = ADNI_info{idx}.FLU_BL;
        ADNI_VBM_MIL{MIL_idx}.MMSE_BL   = ADNI_info{idx}.MMSE_BL;
        ADNI_VBM_MIL{MIL_idx}.RAVLT_BL  = ADNI_info{idx}.RAVLT_BL;
        ADNI_VBM_MIL{MIL_idx}.TRAILS_BL = ADNI_info{idx}.TRAILS_BL;
                            
        if sum(isnan(ADNI_info{idx}.VBM_M6))==0 &&  sum(isempty(ADNI_info{idx}.VBM_M6))==0
            ADNI_VBM_MIL{MIL_idx}.Instance  = [ADNI_VBM_MIL{MIL_idx}.Instance;ADNI_info{idx}.VBM_M6];
        end
        
        if sum(isnan(ADNI_info{idx}.VBM_M12))==0 &&  sum(isempty(ADNI_info{idx}.VBM_M12))==0
            ADNI_VBM_MIL{MIL_idx}.Instance  = [ADNI_VBM_MIL{MIL_idx}.Instance;ADNI_info{idx}.VBM_M12];
        end
        
        if sum(isnan(ADNI_info{idx}.VBM_M18))==0 &&  sum(isempty(ADNI_info{idx}.VBM_M18))==0
            ADNI_VBM_MIL{MIL_idx}.Instance  = [ADNI_VBM_MIL{MIL_idx}.Instance;ADNI_info{idx}.VBM_M18];
        end
        
        if sum(isnan(ADNI_info{idx}.VBM_M24))==0 &&  sum(isempty(ADNI_info{idx}.VBM_M24))==0
            ADNI_VBM_MIL{MIL_idx}.Instance  = [ADNI_VBM_MIL{MIL_idx}.Instance;ADNI_info{idx}.VBM_M24];
        end
        
        if sum(isnan(ADNI_info{idx}.VBM_M36))==0 &&  sum(isempty(ADNI_info{idx}.VBM_M36))==0
            ADNI_VBM_MIL{MIL_idx}.Instance  = [ADNI_VBM_MIL{MIL_idx}.Instance;ADNI_info{idx}.VBM_M36];
        end
        
        if sum(isempty(ADNI_VBM_MIL{MIL_idx}.Instance))==0 && size(ADNI_VBM_MIL{MIL_idx}.Instance,1)>2
            MIL_idx = MIL_idx + 1;
        end
        
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Data Normalization
Dim = size(ADNI_VBM_MIL{1}.Base,2);
X_base = [];
for idx =  1 : size(ADNI_VBM_MIL,2)-1
    X_base = [X_base;ADNI_VBM_MIL{idx}.Base];
end
X_base_mean = mean(X_base,1);
X_max = max(max(abs(X_base)));
for idx =  1 : size(ADNI_VBM_MIL,2)-1
    ADNI_VBM_MIL{idx}.Base = (ADNI_VBM_MIL{idx}.Base - X_base_mean)/X_max;
    ADNI_VBM_MIL{idx}.Instance = (ADNI_VBM_MIL{idx}.Instance - kron(ones(size(ADNI_VBM_MIL{idx}.Instance,1),1),X_base_mean))/X_max;
end    
X_base = (X_base - kron(ones(size(X_base,1),1),X_base_mean))/X_max;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
alpha = 10^-1;
beta  = 10^-2;
Reduce_dim_para =7;
p_para = [0.3,0.5,0.8,1.0,1.5,2];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
Dim_r = Reduce_dim_para* 10;
rho = 1.5;
mu = 2;
p = p_para(1);
W = randn (Dim_r,Dim); W = (W*W')^(-1/2)*W; W = W'; W_base{1} = W; 
W = randn (Dim_r,Dim); W = (W*W')^(-1/2)*W; W = W'; P_base{1} = W;
Lambda_base{1} = randn (Dim,Dim_r);
for idx = 1 : size(ADNI_VBM_MIL,2) - 1
    W = randn (Dim_r,Dim); W = (W*W')^(-1/2)*W; W = W';
    W_instance{idx,1} = W;
    W = randn (Dim_r,Dim); W = (W*W')^(-1/2)*W; W = W';
    P_instance{idx,1} = W; 
    Lambda_instance{idx,1} = randn (Dim,Dim_r);
    X_instance{idx} = ADNI_VBM_MIL{idx}.Instance;
    S{idx,1} = squareform(pdist(ADNI_VBM_MIL{idx}.Instance));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Optimization
for step = 1 : 50
    
    %% Caculate Gamma & theta
    for idx = 1 : size(ADNI_VBM_MIL,2) - 1
        Gamma{step}(idx,idx) = p/2*(sum( (abs(X_base(idx,:)'-W_base{step}*W_base{step}'*X_base(idx,:)')).^p+ eps))^(p/2-1);
        for theta_j = 1 : size(ADNI_VBM_MIL{idx}.Instance,1)
            for theta_k = 1 : size(ADNI_VBM_MIL{idx}.Instance,1)
                theta{idx,step}(theta_j,theta_k) =  p/2*(sum( (abs(W_instance{idx,step}'*(X_instance{idx}(theta_j,:)'...
                    -X_instance{idx}(theta_k,:)'))).^p+eps))^(p/2-1);
            end
        end     
    end
    
    for idx = 1 : size(ADNI_VBM_MIL,2) - 1
        S_ti{idx,step} = S{idx,1} .* theta{idx,step};
        D{idx,step} = diag(sum(S_ti{idx,step}));
        L{idx,step} = D{idx,step}-S_ti{idx,step};
    end
    %% Update W_i
    for idx = 1 : size(ADNI_VBM_MIL,2) - 1
        W_instance{idx,step+1} =  (2*alpha*X_instance{idx}'*L{idx,step}*X_instance{idx}+mu* eye(Dim))^(-1)...
            *(mu*P_instance{idx,step} - Lambda_instance{idx,step});
        
    end
    
    % Update P_i
    for idx = 1 : size(ADNI_VBM_MIL,2) - 1
        N{idx,step} = 2*beta*P_base{step} + mu*W_instance{idx,step} + Lambda_instance{idx,step};
        [U{idx,step},~,V{idx,step}] = svd (N{idx,step},'econ');
        P_instance{idx,step+1} = U{idx,step}*V{idx,step};
    end
    
    %% Update W
    W_base{step+1}= (mu* eye(Dim)-2*X_base'*Gamma{step}*X_base)^(-1)*(mu*P_base{step}-Lambda_base{step});
    
    %% Update P
    N_base{step} = mu*W_base{step+1} + Lambda_base{step};
    for idx = 1 : size(ADNI_VBM_MIL,2) - 1
        N_base{step} = N_base{step} + P_instance{idx,step+1};
    end
    [U_base{step},~,V_base{step}] = svd (N_base{step},'econ');
    P_base{step+1} = U_base{step}*V_base{step};
    
    
    %% Update Lambda & mu
    Lambda_base{step+1}= Lambda_base{step} + mu*(W_base{step+1} - P_base{step} );
    for idx = 1 : size(ADNI_VBM_MIL,2) - 1
        Lambda_instance{idx,step+1} = Lambda_instance{idx,step} + mu * (W_instance{idx,step+1}-P_instance{idx,step});
    end    
    mu = rho * mu;
    
%     obj(step) = -trace (W_base{step+1}'*X_base'*Gamma{step}*X_base*W_base{step+1});
%     obj(step) = obj(step) + mu/2*(norm(W_base{step+1}-P_base{step+1}+W_base{step+1}/mu,'fro'))^2;
%     for idx = 1 : size(ADNI_VBM_MIL,2) - 1
%         obj(step) = obj(step)+alpha*trace(W_instance{idx,step+1}'*X_instance{idx}'*L{idx,step}...
%             *X_instance{idx}*W_instance{idx,step+1})+ beta*(norm(P_base{step+1}-P_instance{idx,step+1},'fro'))^2+...
%             mu/2*(norm(W_instance{idx,step+1}-P_instance{idx,step+1}+Lambda_instance{idx,step+1}/mu,'fro'))^2;
%     end    

    for idx = 1 : size(ADNI_VBM_MIL,2) - 1
        obj(step,1) = sum((abs(X_base(idx,:)'-W_base{step+1}*W_base{step+1}'*X_base(idx,:)')).^p);
        obj(step,2) = beta*(norm(W_base{step+1}-W_instance{idx,step+1},'fro'))^2;
        obj(step,3) = 0;
        for theta_j = 1 : size(ADNI_VBM_MIL{idx}.Instance,1)
            for theta_k = 1 : size(ADNI_VBM_MIL{idx}.Instance,1)
                obj(step,3) =  obj(step,3) + alpha * sum((abs(W_instance{idx,step+1}'*(X_instance{idx}(theta_j,:)'...
                    -X_instance{idx}(theta_k,:)'))).^p);
            end
        end
    end    
    
    diff_W_P(step) = norm(W_base{step+1}'-P_base{step+1}','fro');
    for idx = 1 : size(ADNI_VBM_MIL,2) - 1
        diff_W_P(step) = diff_W_P(step) + norm(W_instance{idx,step+1}-P_instance{idx,step+1},'fro');
    end    
    
    
    
end
J = sum(obj,2);

%%