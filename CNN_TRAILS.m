close all
clc
clear

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
    X_TRAILS{idx}=[];
end
for idx = 1 : size(ADNI_VBM_MIL,2)-1
if sum(isnan(ADNI_VBM_MIL{idx}.TRAILS_BL))==0 && sum(isempty(ADNI_VBM_MIL{idx}.TRAILS_BL))==0 ...
     && sum(sum(isnan(W_PCA_KNN{1,idx})))==0  && sum(sum(isnan(W_PCA_KNN{2,idx})))==0 ...
     && sum(sum(isnan(W_PCA_KNN{3,idx})))==0  && sum(sum(isnan(W_PCA_KNN{4,idx})))==0 ...
     && sum(sum(isnan(W_PCA_KNN{5,idx})))==0
    X_idx = 1;
    for Dim_r_idx = 1 : size(W_para,1)
    for p_para_idx = 1 : size(W_para,2)
       X_TRAILS{X_idx} = [X_TRAILS{X_idx};ADNI_VBM_MIL{idx}.Base*W_para{Dim_r_idx,p_para_idx,idx}];
       X_idx = X_idx + 1;
    end
    end
    
    for Dim_r_idx = 1 : size(W_para,1)
       X_TRAILS{X_idx} = [X_TRAILS{X_idx};ADNI_VBM_MIL{idx}.Base*W_PCA_KNN{Dim_r_idx,idx}'];
       X_idx = X_idx + 1;        
    end    
    X_TRAILS{X_idx} = [X_TRAILS{X_idx};ADNI_VBM_MIL{idx}.TRAILS_BL];
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
KFold_Num = 5;
CV_Indice = crossvalind('kfold',size(X_TRAILS{X_idx},1),KFold_Num );
for CV_idx = 1 : KFold_Num
    for X_TRAILS_idx = 1 : size(X_TRAILS,2)
        X_train_CV{X_TRAILS_idx} = [];X_test_CV{X_TRAILS_idx} = [];
        for idx = 1 : size(X_TRAILS{4},1)
            if CV_Indice(idx) ~= CV_idx
                X_train_CV{X_TRAILS_idx} = [X_train_CV{X_TRAILS_idx};X_TRAILS{X_TRAILS_idx}(idx,:)];
            else
                X_test_CV{X_TRAILS_idx}  = [X_test_CV{X_TRAILS_idx};X_TRAILS{X_TRAILS_idx}(idx,:)];
                
            end
        end
    end
    
 for Mdl_idx = 1 : size(X_TRAILS,2)-1
        X_train = X_train_CV{Mdl_idx}; Y_train = X_train_CV{size(X_TRAILS,2)};
        X_test  = X_test_CV{Mdl_idx};  Y_test  = X_test_CV{size(X_TRAILS,2)};
     for DX_idx = 1 : size(Y_train,2)

        CNN_Training_data = reshape(X_train',[1,size(X_train,2),1,size(X_train,1)]);
        CNN_Label_data = reshape(Y_train(:,DX_idx)',[1,1,1,size(Y_train,1)] );
        CNN_Test_data = reshape(X_test',[1,size(X_test,2),1,size(X_test,1)]);
        layers = [
            imageInputLayer([1 size(CNN_Training_data,2) 1])                
            convolution2dLayer([1 5], 16)
            reluLayer
            convolution2dLayer([1 10], 32)
            reluLayer
            maxPooling2dLayer([1 2])            
            dropoutLayer(0.3)
            fullyConnectedLayer(1)
            regressionLayer];

        options = trainingOptions('sgdm', ...
            'MiniBatchSize',16, ...
            'MaxEpochs',30, ...
            'InitialLearnRate',1e-4, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropFactor',0.2, ...
            'LearnRateDropPeriod',10, ...
            'Verbose',0);
        net = trainNetwork(CNN_Training_data,CNN_Label_data,layers,options);
        YPredicted = predict(net,CNN_Test_data); 
        RMSE_CNN(Mdl_idx,DX_idx,CV_idx) = sqrt(mean(Y_test(:,DX_idx)-YPredicted).^2);           
     end
 end    
end

 RMSE_CNN_TRAILS_avg = mean(RMSE_CNN,3);
 Std_CNN_TRAILS      = std(RMSE_CNN,0,3);
 save('CNN_VBM_TRAILS.mat','RMSE_CNN_TRAILS_avg','Std_CNN_TRAILS');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%