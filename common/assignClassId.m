function labels = assignClassId(Y, class_ids)
% assign labels in CLASS_IDS for Y
% Y: n*1
% CLASS_IDS: c*1
labels = zeros(size(Y));
for i = 1:length(class_ids)
    d(:,i) = abs(Y-class_ids(i));
end
[~,min_ind] = min(d,[],2);
labels = class_ids(min_ind);
end