function AP = computeAveragePrecision(labels, predict_labels)
% compute average precision for multi-class classification problem
classid = unique(labels);
precision = zeros(1,length(classid));
for i = 1:length(classid)
    ind = find(labels==classid(i));
    precision(i) = sum(labels(ind)==predict_labels(ind))/length(ind);
end
AP = mean(precision);
end