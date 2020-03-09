function labels_predict = MSLRDA_MVSRC_predict(testDataC, data_dict, labels_dict, options)
% 
testDataC = preprocessMultiModalityData(testDataC, [], options, options.bExtended);

data_tst = cell2mat(testDataC);
data_tst = data_tst';

view_sizes = zeros(1,size(testDataC,2));
for view_i = 1:size(testDataC,2)
    view_sizes(view_i) = size(testDataC{1,view_i},2);
end

class_id = unique(labels_dict);
labels_predict = zeros(size(data_tst,2),1);

for sub_i = 1:size(data_tst,2)
    query = data_tst(:,sub_i);
    A = MVSRC(query, data_dict, view_sizes, options.eta, options.phi,options.max_iter);
    view_n = length(view_sizes);
    loss_class = zeros(1,length(class_id));
    if options.display
        fprintf('%d:', sub_i);
    end
    for class_i = 1:length(class_id)
        ind_class = labels_dict==class_id(class_i);
        for view_i = 1:view_n
            if view_i ==1
                ind1_viewi = 1;
            else
                ind1_viewi = sum(view_sizes(1:view_i-1)+1);
            end
            ind2_viewi = sum(view_sizes(1:view_i));
            q_viewi = query(ind1_viewi:ind2_viewi,:);
            data_viewi_class = data_dict(ind1_viewi:ind2_viewi,ind_class);
            alpha_viewi_class = A(ind_class,view_i);
            diff_viewi1 = q_viewi-data_viewi_class*alpha_viewi_class;
            loss_class(class_i) = loss_class(class_i)+diff_viewi1'*diff_viewi1;
        end
        if options.display
            fprintf('%f\t',loss_class(class_i));
        end
    end
    [~, ind_min] = min(loss_class);
    labels_predict(sub_i) = class_id(ind_min);
    if options.display
        fprintf('%d\n',labels_predict(sub_i));
    end
end
end