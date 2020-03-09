function [out,flags, Mus, Sigmas] = preprocessMultiModalityData(data, labels, options, bExtend)
% preprocess multiple modality data for multiple task learning
% DATA: data to be processed
% LABELS: labels for data
% options: parameters for preprocessing.
% OUT: the output data after DATA being processed
% FLAGS: feature selection vectors for each modality of DATA.

% 2016.9.16 UNC-CH
% 2016.12.22 纠正了当length(label_set)==1时的bug，zscore(data_viewi)改为zscore(data_viewi,0,1);;
% 2016.8.6 TTEST,如果大于两类，选取最大的两类进行特征选取

[task_n,view_n] = size(data);
task_size = zeros(1,task_n);
for task_i = 1:task_n
    task_size(task_i) = size(data{task_i,1},1);
end
Ys = cell2mat(labels);

for view_i = 1:view_n
    data_viewi = cell2mat(data(:,view_i));
    [data_viewi, flags{view_i}] = featureSelection(data_viewi, Ys, options.feaSelMethod, options.feaSelNum(view_i),options.feaSelFlags{view_i});
    [data_viewi, Mus{view_i}, Sigmas{view_i}] = zscore(data_viewi,0,1);


    finish = 0;
    for task_i = 1:task_n
        start = finish+1;
        finish = finish + task_size(task_i);
        out{task_i,view_i} = data_viewi(start:finish,:);
    end
end

if (nargin == 4 & bExtend==0)
   return; 
end

for view_i = 1:view_n
    for task_i = 1:task_n
        out{task_i,view_i} = [out{task_i,view_i} ones(task_size(task_i),1)];
    end
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
                if length(label_set) == 1
                    % using CORR to select features
                    cols_summary = zeros(1,size(data,2));
                    for dim_i = 1:size(data,2)
                        cols_summary(dim_i) = abs(corr(labels+eps*rand(size(labels)), data(:,dim_i)+eps*rand(size(labels))));
                    end
                else% length(label_set)>= 2
                    % 2016.11.22, 2017.8.6
                    class_sizes = zeros(1,length(label_set));
                    for class_i = 1:length(label_set)
                        class_sizes(class_i) = sum(labels==label_set(class_i));
                    end
                    [~,indices] = sort(class_sizes,'descend');
                    ind1 = (labels==label_set(indices(1)));
                    ind2 = (labels==label_set(indices(2)));
%                     ind1 = (labels==label_set(1));
%                     ind2 = (labels==label_set(2));
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
                    cols_summary = 1-p;
%                 else % length(label_set)>2
                    % 2016.8.6
%                     wp = zeros(nchoosek(length(label_set),2),size(data,2));
%                     k = 1;
%                     for class_i = 1:length(label_set)
%                         for class_j = class_i+1:length(label_set)
%                             ind1 = (labels==label_set(class_i));
%                             w1 = sum(ind1)/length(labels);
%                             ind2 = (labels==label_set(class_j));
%                             w2 = sum(ind2)/length(labels);
%                             if sum(ind1) > 1
%                                 data1 = data(ind1,:);
%                             else
%                                 data1 = [data(ind1,:); data(ind1,:)];
%                             end
%                             if sum(ind2) > 1
%                                 data2 = data(ind2,:);
%                             else
%                                 data2 = [data(ind2,:); data(ind2,:)];
%                             end
%                             [~,p_value] = ttest2(data1,data2);
%                             wp(k,:) = w1*w2*p_value;
%                             k = k+1;
%                         end
%                     end
%                     cols_summary = 1-sum(wp,1)/nchoosek(length(label_set),2);
                end
             case 'CORR'
                cols_summary = zeros(1,size(data,2));
                for dim_i = 1:size(data,2)
                    cols_summary(dim_i) = abs(corr(labels, data(:,dim_i)+1e-6));
                end
            otherwise
                ;
        end
        nan_ind = isnan(cols_summary);
        cols_summary(nan_ind) = 0;
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