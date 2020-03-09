function [feaFalgs, data_dict, labels_dict] = MVSRC_train(trainDataC,trainLabelsC, dict_size, class_id, options)
% generate training data for MVSRC. 
% 2018.8.26 @ Jiangnan University, Wuxi
[trainDataC, feaFalgs] = preprocessMultiModalityData(trainDataC,trainLabelsC,options, options.bExtended);

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

[data_dict, labels_dict] = generateDictionary(data_trn, labels_trn, dict_size, class_id);
end