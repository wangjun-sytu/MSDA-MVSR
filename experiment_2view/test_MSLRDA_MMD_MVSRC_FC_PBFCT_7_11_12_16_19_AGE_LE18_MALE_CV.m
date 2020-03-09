% small dataset
clear

warning off; 

ts = getTimeStamp();

ds=load('fMRI_FC_PBFCT_MultiClass_AGE_LE18_MALE.mat');
% 'CALTECH'(1),'CMU'(2),'KKI'(3),'LEUVEN_1'(4),'LEUVEN_2'(5),'MAX_MUN'(6),'NYU'(7),'OHSU'(8),'OLIN'(9),'PITT'(10)
% 'SBL'(11),'SDSU'(12),'STANFORD'(13),'UCLA_1'(14),'UCLA_2'(15),'UM_1'(16),'UM_2'(17),'USM'(18),'YALE'(19)
center_list = [7 11 12 16 19]; 
class_id = [0 1 2];
trainOptions.maxClassSizes = [10, 20, 10, 10, 10];

data_all = [cell2mat(ds.FC(center_list)') cell2mat(ds.PBFA(center_list)')]';
labels_all = cell2mat(ds.Y_DSM_IV(center_list)');
Indices4CV_all = ds.Indices4CV(center_list);

% remove subjects with invalid labels
task_sizes = zeros(1,length(ds.FC(center_list)));
for task_i = 1:length(ds.FC(center_list))
    task_sizes(task_i) = size(ds.FC{center_list(task_i)},1);
end
valid_ind = (labels_all~=-1 & labels_all~=3);
valid_ind_C = mat2cell(valid_ind,task_sizes,1);
data_all = data_all(:,valid_ind);
labels_all = labels_all(valid_ind);
for task_i = 1:length(Indices4CV_all)
    Indices4CV_all{task_i} = Indices4CV_all{task_i}(valid_ind_C{task_i});
    task_sizes(task_i) = length(Indices4CV_all{task_i});
end

view_sizes_0 = [6670, 359]; % the input data
view_sizes = [200 50];     % after feature selection
data_all_C = mat2cell(data_all', task_sizes, view_sizes_0);
labels_all_C = mat2cell(labels_all, task_sizes, 1);

trainOptions.feaSelMethod = 'TTEST';
trainOptions.feaSelNum = view_sizes;
trainOptions.feaSelFlags{1} = [];
trainOptions.feaSelFlags{2} = [];
trainOptions.bExtended = 0;

task_n = length(task_sizes);
trainOptions.eta=1e-10;
trainOptions.phi = 1;
trainOptions.max_iter = 1000;
trainOptions.layer_n = 5;
trainOptions.betas = [0.4 0.6];
trainOptions.kernelType = 'linear';

labels_predict = -ones(length(labels_all_C{end}),1);
labels_gt = labels_all_C{end};

for fold_i = 1:length(unique(Indices4CV_all{end}))
    % split the input data into training and testing sets
    trainLabelsC = cell(size(data_all_C,1),1);
    testLabelsC = cell(size(data_all_C,1),1);
    fold_id_test = fold_i;
%     fold_id_test = [2*fold_i-1 2*fold_i];
    for center_i = 1:length(center_list)
        [trainData, trainLabels, testData, testLabels, testIndices] = splitData(cell2mat(data_all_C(center_i,:)), labels_all_C{center_i}, Indices4CV_all{center_i}, fold_id_test);
        task_sizes_trn(center_i) = size(trainData,1);
        task_sizes_tst(center_i) = size(testData,1);
        trainDataC(center_i,:) = mat2cell(trainData, size(trainData,1), view_sizes_0);
        testDataC(center_i,:) = mat2cell(testData, size(testData,1), view_sizes_0);
        trainLabelsC{center_i} = trainLabels;
        testLabelsC{center_i} = testLabels;
    end

    trainOptions.tau = [];
    trainOptions.lambda = [];
    trainOptions.tauCandidates = [1e-3 0.01 0.1 0 1 10 100 1e3 1e4];
    trainOptions.lambdaCandidates = [1e-3 0.01 0.1 0 1 10 100 1e3 1e4];

    [trainDataC, trainLabelsC] = setClassSize(trainDataC, trainLabelsC, trainOptions.maxClassSizes);
    
    testClassid = unique(testLabelsC{end});
    dict_size = zeros(1, length(class_id));
    for class_i = 1:length(testClassid)
        dict_size(class_id==testClassid(class_i)) = sum(cell2mat(trainLabelsC)==testClassid(class_i));
    end
    
    [feaFlags, data_dict, labels_dict, optimal_params] = MSLRDA_MMD_MVSRC_train(trainDataC,trainLabelsC, dict_size, class_id, trainOptions);
    trainOptions.feaSelFlags = feaFlags;

    % test
    predictOptions = trainOptions;
    predictOptions.feaSelFlags = feaFlags;
    predictOptions.feaSelMethod = [];
    predictOptions.display = true;

    labels_predict(testIndices) = MSLRDA_MVSRC_predict(testDataC(end,:), data_dict, labels_dict, predictOptions);
end

AveragePrecision = computeAveragePrecision(labels_gt, labels_predict);

for class_i = 1:length(testClassid)
    id = testClassid(class_i);
    CorrectNums(class_i) = sum(labels_gt(labels_gt==id) == labels_predict(labels_gt==id));
end

for class_i = 1:length(testClassid)
    id = testClassid(class_i);
    fprintf('%d:%d (%d)\n', id, CorrectNums(class_i),sum(labels_gt==id));
end
fprintf('Average Precision: %.4f\n', AveragePrecision);
CP = classperf(labels_gt, labels_predict);

cmd = 'test_MSLRDA_MMD_MVSRC_FC_PBFCT_7_11_12_16_19_AGE_LE18_MALE_CV';
savefile = ['MSLRDA_MMD_MVSRC_FC_PBFCT_MultiClass_7_11_12_16_19_AGE_LE18_MALE_' ts '.mat'];
save(savefile, 'center_list', 'class_id', 'trainOptions','predictOptions', 'CorrectNums', 'AveragePrecision', 'cmd');
