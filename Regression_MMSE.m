clear
clc
close all

Reduce_dim_para = (5:9);
p_para = [0.3,0.5,0.8,1.0,1.5,2];

load('PCA_LPP_VBM_Para_Exp.mat')

load('ADNI_VBM_MIL.mat')
for idx = 1 : size(ADNI_VBM_MIL,2)-1
    for Reduce_dim_para=5:9
     dim = 10*Reduce_dim_para;
    W_PCA_KNN{Reduce_dim_para-4,idx} = ADNI_VBM_MIL{idx}.Weight{dim/10};
    end
end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Data Generation
for idx = 1 : size(W_para,1)*size(W_para,2)+size(W_PCA_KNN,1)+1
    X_MMSE{idx}=[];
end
for idx = 1 : size(ADNI_VBM_MIL,2)-1
if sum(isnan(ADNI_VBM_MIL{idx}.MMSE_BL))==0 && sum(isempty(ADNI_VBM_MIL{idx}.MMSE_BL))==0 ...
     && sum(sum(isnan(W_PCA_KNN{1,idx})))==0  && sum(sum(isnan(W_PCA_KNN{2,idx})))==0 ...
     && sum(sum(isnan(W_PCA_KNN{3,idx})))==0  && sum(sum(isnan(W_PCA_KNN{4,idx})))==0 ...
     && sum(sum(isnan(W_PCA_KNN{5,idx})))==0
    X_idx = 1;
    for Dim_r_idx = 1 : size(W_para,1)
    for p_para_idx = 1 : size(W_para,2)
       X_MMSE{X_idx} = [X_MMSE{X_idx};ADNI_VBM_MIL{idx}.Base*W_para{Dim_r_idx,p_para_idx,idx}];
       X_idx = X_idx + 1;
    end
    end
    
    for Dim_r_idx = 1 : size(W_para,1)
       X_MMSE{X_idx} = [X_MMSE{X_idx};ADNI_VBM_MIL{idx}.Base*W_PCA_KNN{Dim_r_idx,idx}'];
       X_idx = X_idx + 1;        
    end    
    X_MMSE{X_idx} = [X_MMSE{X_idx};ADNI_VBM_MIL{idx}.MMSE_BL];
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
KFold_Num = 5;
CV_Indice = crossvalind('kfold',size(X_MMSE{X_idx},1),KFold_Num );
for CV_idx = 1 : KFold_Num
    for X_MMSE_idx = 1 : size(X_MMSE,2)
        X_train{X_MMSE_idx} = [];X_test{X_MMSE_idx} = [];
        for idx = 1 : size(X_MMSE{4},1)
            if CV_Indice(idx) ~= CV_idx
                X_train{X_MMSE_idx} = [X_train{X_MMSE_idx};X_MMSE{X_MMSE_idx}(idx,:)];
            else
                X_test{X_MMSE_idx}  = [X_test{X_MMSE_idx};X_MMSE{X_MMSE_idx}(idx,:)];
                
            end
        end
    end
    
   for Mdl_idx = 1 : size(X_MMSE,2)-1
        Mdl_SVR{Mdl_idx,CV_idx}=fitrsvm(X_train{Mdl_idx},X_train{(size(X_MMSE,2))},...
            'Standardize',true,'KernelFunction','gaussian');
        Y_predict_SVR{Mdl_idx,CV_idx} = predict(Mdl_SVR{Mdl_idx,CV_idx},X_test{Mdl_idx});
        RMSE_SVR(Mdl_idx,CV_idx) = sqrt(mean(X_test{(size(X_MMSE,2))}-Y_predict_SVR{Mdl_idx,CV_idx}).^2);
        
        Mdl_RR{Mdl_idx,CV_idx}=fitrlinear(X_train{Mdl_idx},X_train{(size(X_MMSE,2))},'Lambda',10.^(-5:5),'Regularization','ridge');
        Y_RR = predict(Mdl_RR{Mdl_idx,CV_idx},X_test{Mdl_idx});
        Y_predict_RR{Mdl_idx,CV_idx} = min(Y_RR,[],2);
        RMSE_RR(Mdl_idx,CV_idx) = sqrt(mean(X_test{(size(X_MMSE,2))}-Y_predict_RR{Mdl_idx,CV_idx}).^2);

        Mdl_LR{Mdl_idx,CV_idx}=fitrlinear(X_train{Mdl_idx},X_train{(size(X_MMSE,2))},'Lambda',0);
        Y_predict_LR{Mdl_idx,CV_idx} = predict(Mdl_LR{Mdl_idx,CV_idx},X_test{Mdl_idx});
        RMSE_LR(Mdl_idx,CV_idx) = sqrt(mean(X_test{(size(X_MMSE,2))}-Y_predict_LR{Mdl_idx,CV_idx}).^2);        
        
        Mdl_Lasso{Mdl_idx,CV_idx}=fitrlinear(X_train{Mdl_idx},X_train{(size(X_MMSE,2))},'Lambda',10.^(-5:5),'Regularization','lasso');
        Y_Lasso = predict(Mdl_Lasso{Mdl_idx,CV_idx},X_test{Mdl_idx});
        Y_predict_Lasso{Mdl_idx,CV_idx} = min(Y_Lasso,[],2);
        RMSE_Lasso(Mdl_idx,CV_idx) = sqrt(mean(X_test{(size(X_MMSE,2))}-Y_predict_Lasso{Mdl_idx,CV_idx}).^2);
        
   end  
end


    RMSE_SVR_avg_MMSE = mean(RMSE_SVR,2);
    Std_SVR_avg_MMSE  = std(RMSE_SVR,0,2);
    RMSE_LR_avg_MMSE  = mean(RMSE_LR,2);
    Std_LR_avg_MMSE   = std(RMSE_LR,0,2);
    RMSE_RR_avg_MMSE  = mean(RMSE_RR,2);
    Std_RR_avg_MMSE   = std(RMSE_RR,0,2);
    RMSE_Lasso_avg_MMSE = mean(RMSE_Lasso,2);
    Std_Lasso_avg_MMSE  = std(RMSE_Lasso,0,2);

    save('R_VBM_MMSE.mat','RMSE_SVR_avg_MMSE','RMSE_LR_avg_MMSE','RMSE_RR_avg_MMSE','RMSE_Lasso_avg_MMSE',...
          'Std_SVR_avg_MMSE','Std_LR_avg_MMSE','Std_RR_avg_MMSE','Std_Lasso_avg_MMSE',...
          'RMSE_SVR','RMSE_LR','RMSE_RR','RMSE_Lasso');
