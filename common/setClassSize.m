function [dataC_out, labelsC_out] = setClassSize(dataC_in, labelsC_in, max_class_sizes)
% set the class size 
% max_class_sizes: max class sizes for each task.
[task_n, view_n] = size(dataC_in);
dataC_out = cell(task_n, view_n);
labelsC_out = cell(task_n,1);
for task_i = 1:task_n   
    for view_i = 1:view_n
        [dataC_out{task_i,view_i}, labelsC_out{task_i}] = adjust(dataC_in{task_i,view_i}, labelsC_in{task_i},max_class_sizes(task_i));
    end
end
end

function [data_out, labels_out] = adjust(data_in, labels_in, max_class_size)
% adjust the sizes of datasets
data_out = zeros(size(data_in));
labels_out = -ones(size(labels_in));
classids = unique(labels_in);
for class_i = 1:length(classids)
    indices = find(labels_in==classids(class_i));
    if length(indices)> max_class_size
        data_out(indices(1:max_class_size),:) = data_in(indices(1:max_class_size),:);
        labels_out(indices(1:max_class_size)) = labels_in(indices(1:max_class_size));
    else
        data_out(indices,:) = data_in(indices,:);
        labels_out(indices) = labels_in(indices);
    end
end
valid_ind = labels_out>=0;
data_out(~valid_ind,:) = [];
labels_out(~valid_ind) = [];
end