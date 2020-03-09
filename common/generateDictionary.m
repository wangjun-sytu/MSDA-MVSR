function [data_dict, labels_dict] = generateDictionary(data, labels, dict_size, classid)
% generate dictionary from data.
% data: DxN
% labels: the corresponding labels for data
% dict_size: the final dictionary size. 1*C
% labels_dict: the labels for data_dict.
% 2017.8.7 by Jun Wang


data_dict = [];
labels_dict = [];
for class_i = 1:length(classid)
    data_classi = data(:,labels==classid(class_i));
    labels_classi = labels(labels==classid(class_i));
    
%     ind_rp = randperm(length(labels_classi));
    class_size = min(dict_size(class_i),sum(labels==classid(class_i)));
    data_dict = [data_dict data_classi(:,1:class_size)];
    labels_dict = [labels_dict; labels_classi(1:class_size)];
%     data_dict = [data_dict data_classi(:,ind_rp(1:class_size))];
%     labels_dict = [labels_dict; labels_classi(ind_rp(1:class_size))];
end
end