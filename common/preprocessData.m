function [out,flags, MRI_mu, MRI_sigma, fMRI_mu, fMRI_sigma] = preprocessData(data, labels, options, bExtend)
% preprocess multiple modality data for multiple task learning
% DATA: data to be processed
% LABELS: labels for data
% options: parameters for preprocessing.
% OUT: the output data after DATA being processed
% FLAGS: feature selection vectors for each modality of DATA.

% 2016.4.5 UNC-CH
% 2016.4.22 add MRI_MU, MRI_SIGMA, fMRI_MU, fMRI_SIGMA 
% 2016.6.13 add 'TTEST' to FEATURESELECTION
% 2016.11.21 修正了featureSelection()的一个bug.

[task_n,view_n] = size(data);
task_size = zeros(view_n,task_n);
Ys = cell2mat(labels);

% MRI
MRI_data = [];
for task_i = 1:task_n
    task_size(task_i,1) = size(data{task_i,1},1);
    MRI_data = [MRI_data; data{task_i,1}];
end

[MRI_data, flags{1}] = featureSelection(MRI_data, Ys, options.feaSelMethod, options.feaSelNum(1),options.feaSelFlags{1});

[MRI_data, MRI_mu, MRI_sigma] = zscore(MRI_data);

finish = 0;
for task_i = 1:task_n
    start = finish+1;
    finish = finish + task_size(task_i,1);
    out{task_i,1} = MRI_data(start:finish,:);
end



% fMRI
fMRI_data = [];
for task_i = 1:task_n
    task_size(task_i,2) = size(data{task_i,2},1);
    fMRI_data = [fMRI_data; data{task_i,2}];
end

[fMRI_data, flags{2}] = featureSelection(fMRI_data, Ys, options.feaSelMethod, options.feaSelNum(2),options.feaSelFlags{2});

[fMRI_data, fMRI_mu, fMRI_sigma] = zscore(fMRI_data);

finish = 0;
for task_i = 1:task_n
    start = finish+1;
    finish = finish + task_size(task_i,2);
    out{task_i,2} = fMRI_data(start:finish,:);
end

if (nargin == 4 & bExtend==0)
   return; 
end
    
for task_i = 1:task_n
    out{task_i,1} = [out{task_i,1} ones(task_size(task_i,1),1)];
    out{task_i,2} = [out{task_i,2} ones(task_size(task_i,2),1)];
end
end

function [data,flags] = featureSelection(data, labels, feaSelMethod, feaSelNum, feaSelFlags)
% Select features according to feaSelNum and feaSelFlags
[data_n,dim_n] = size(data);
if feaSelNum<size(data,2)
    if isempty(feaSelFlags)
        switch feaSelMethod
            case 'VAR'
                cols_summary = var(data);
            case 'NORM0'
                cols_summary = zeros(1,size(data,2));
                for dim_i = 1:dim_n
                    ind = find(data(:,dim_i)~=0);
                    cols_summary(dim_i) = length(ind);
                end
            case 'MEAN'
                cols_summary = zeros(1,size(data,2));
                label_set = unique(labels);
                for dim_i = 1:dim_n
                    ind1 = (labels==label_set(1));
                    mean1 = mean(data(ind1,dim_i));
                    ind2 = (labels==label_set(2));
                    mean2 = mean(data(ind2,dim_i));
                    cols_summary(dim_i) = abs(mean1-mean2);
                end
            case 'TTEST'
                label_set = unique(labels);
                cols_summary = zeros(1,dim_n);
                % 2016.11.21 
                ind1 = (labels==label_set(1));
                ind2 = (labels==label_set(2));
                if sum(ind1) > 1
                    data1 = data(ind1,:);
                else
                    data1 = [data(ind1,:); data(ind1,:)];
                end
                if sum(ind2) > 1
                    data2 = data(ind2,:);
                else
                    data2 = [data(ind2,:); data(ind2,:)];
                end
                [~,p] = ttest2(data1,data2);
%                 [~,p] = ttest2(data(ind1,:),data(ind2,:));
                cols_summary = 1-p;
             case 'CORR'
                cols_summary = zeros(1,size(data,2));
                for dim_i = 1:size(data,2)
                    cols_summary(dim_i) = abs(corr(labels, data(:,dim_i)+1e-6));
                end
            otherwise
                ;
        end
        sort_cols = sort(cols_summary,'descend');
        gate = sort_cols(feaSelNum);
        flags = (cols_summary>=gate);
    else
        flags = feaSelFlags;
    end
    data = data(:,flags);
else
    flags = (ones(1,size(data,2))==1);
end
end