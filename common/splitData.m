function [trainData, trainLabels, testData, testLabels, testIndices] = splitData(dataC, labelsC, Indices4CV, testIds)
% split the data according to Indices4CV
% testIds: IDs in Indices4CV for test data
% testIndices: Indices of the test data in input dataC

testIndices = Indices4CV==testIds(1);
for i = 2:length(testIds)
    testIndices = testIndices | (Indices4CV==testIds(i));
end

testData = dataC(testIndices,:);
testLabels = labelsC(testIndices,:);
trainData = dataC(~testIndices,:);
trainLabels = labelsC(~testIndices,:);
end