function [feaFalgs, data_dict, labels_dict, optimal_params] = MSLRDA_MMD_MVSRC_train(trainDataC,trainLabelsC, dict_size, class_id, options)
% optimize 
% 2018.8.20 @ Jiangnan University, Wuxi
% 2019.6.2 fix a bug that MMD includes NaN.
[trainDataC, feaFalgs] = preprocessMultiModalityData(trainDataC,trainLabelsC,options, options.bExtended);

% trainLabels_target = trainLabelsC{end};
% [trainDataC{end,1}, trainLabelsC{end}, dict_size] = balanceData(trainDataC{end,1}, trainLabels_target, dict_size, 10);
% [trainDataC{end,2}, trainLabelsC{end}, dict_size] = balanceData(trainDataC{end,2}, trainLabels_target, dict_size, 10);

task_sizes_trn = zeros(1,size(trainDataC,1));
for center_i = 1:size(trainDataC,1)
    task_sizes_trn(center_i) = size(trainDataC{center_i,1},1);
end

view_sizes = zeros(1,size(trainDataC,2));
for view_i = 1:size(trainDataC,2)
    view_sizes(view_i) = size(trainDataC{1,view_i},2);
end

data_trn = cell2mat(trainDataC); data_trn = data_trn';
labels_trn = cell2mat(trainLabelsC);

minMMD = inf;
for lambda_i = 1:length(options.lambdaCandidates)
    for tau_i = 1:length(options.tauCandidates)
        [~,data_trn_LRR,MMDs] = MSLRDA_MMD(data_trn, labels_trn, task_sizes_trn, view_sizes, options.tauCandidates(tau_i), options.lambdaCandidates(lambda_i));
        MMD_table(lambda_i,tau_i) = sum(MMDs(~isnan(MMDs))); % 2019.6.2
        if MMD_table(lambda_i,tau_i)<minMMD
            data_trn_LRR_optimal = data_trn_LRR;
            optimal_params.lambda = options.lambdaCandidates(lambda_i);
            optimal_params.tau = options.tauCandidates(tau_i);
            minMMD = sum(MMDs);
        end
    end
end

[data_dict, labels_dict] = generateDictionary(data_trn_LRR_optimal, labels_trn, dict_size, class_id);
end