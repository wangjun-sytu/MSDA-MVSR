function [Z,data_LRR, MMD] = MSLRDA_MMD(data, labels, task_sizes, view_sizes, tau, lambda)
% call hidden space discrinative LRR for multiple sources. Obtain MMD for each layer
% data: training samples, DxN
% labels:  labels of the training samples, Nx1;
% A: N*M
% task_sizes: the data numbers in each task
% view_sizes: the feature numbers in each view
% Jun Wang
% 2018.10.29

% trasfer LRR
ind1_task_t = sum(task_sizes(1:end-1))+1;
ind2_task_t = sum(task_sizes);
data_trn = cell(length(view_sizes),1);
data_t = cell(length(view_sizes),1);
Z = cell(length(view_sizes),1);

WM = ~squareform(pdist(labels,'hamming'));
for i = 1:size(WM,1)
    W_LDA(i,:) = WM(i,:)/sum(WM(i,:));
end

LM_LDA = computeLaplacianMatrix(W_LDA);

data_W = cell(length(view_sizes), length(task_sizes));
MMD = zeros(1,length(task_sizes));

for view_i = 1:length(view_sizes)
    ind1_viewi = sum(view_sizes(1:view_i-1))+1;
    ind2_viewi = sum(view_sizes(1:view_i));
    % transform training data into LRR space
    data_trn{view_i} = data(ind1_viewi:ind2_viewi,:);
    Z_init = rand(ind2_task_t-ind1_task_t+1, size(data_trn{view_i},2));
    E_init = rand(size(data_trn{view_i}));
    data_t{view_i} = data_trn{view_i}(:,ind1_task_t:ind2_task_t);
    [W_viewi, ~, E_viewi, iter_viewi] = MSLRDA(data_trn{view_i}, data_t{view_i}, task_sizes, tau, lambda, LM_LDA, Z_init, E_init); 
    data_trn_viewi_C = mat2cell(data_trn{view_i},size(data_trn{view_i},1),task_sizes);
    for task_i = 1:length(task_sizes)
            data_W{view_i,task_i} = W_viewi{task_i}*data_trn_viewi_C{task_i};
    end
end
for task_i = 1:length(task_sizes)
    MMD(task_i) = computeMMD(cell2mat(data_W(:,task_i)),cell2mat(data_W(:,end))); % compute MMD between each source domain and the target domain
end

data_LRR = cell2mat(data_W);
end

function [E] = solve_l1l2(W,tau)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),tau);
end
end

function [W,Z,E,iter] = MSLRDA(S, T, S_group_sizes, tau, lambda, LC, Z_init, E_init)
%multiple sources Low Rank Domain adaptation
% inputs:
%        S -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors.
%        T -- D*M matrix of a dictionary, M is the size of the dictionary
%        S_group_sizes -- group sizes for X

tol = 1e-8;
maxIter = 500;
[d n] = size(S);
m = size(T,2);
rho = 1.1;
max_mu = 1e30;
mu = 1e0;
% inv_a = inv(A'*A+eye(m));
%% Initializing optimization variables
% intialize
Z = Z_init;% zeros(m,n);
E = E_init; % sparse(d,n);

P = zeros(m,n);

Y1 = zeros(d,n);
Y2 = zeros(m,n);
Y3 = zeros(m,n);

%% Start main loop
iter = 0;
% disp(['initial,rank=' num2str(rank(Z))]);
W = cell(1,length(S_group_sizes));
for group_i = 1:length(S_group_sizes)
    W{group_i} = eye(d);
end
X_cell = mat2cell(S,size(S,1),S_group_sizes);
H_cell = X_cell;
while iter<maxIter
    iter = iter + 1;
    %update W
    S_cell = H_cell(1:end-1);
    Z_cell = mat2cell(Z,size(Z,1),S_group_sizes);
    E_cell = mat2cell(E,size(E,1),S_group_sizes);
    Y1_cell = mat2cell(Y1,size(Y1,1),S_group_sizes);
    for group_i = 1:length(S_group_sizes)
        if group_i == length(S_group_sizes)
            W{group_i} = eye(d);
            continue;
        end
        tmp1 = mu*S_cell{group_i}*S_cell{group_i}'+eye(d)*1e-3;
        tmp2 = mu*(T*Z_cell{group_i}+E_cell{group_i})*S_cell{group_i}'-Y1_cell{group_i}*S_cell{group_i}';
        W{group_i} = tmp2/tmp1;
    end
    
    %update J
    temp = Z + Y2/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    %udpate Z
    for group_i = 1:length(S_group_sizes)
        if group_i<length(S_group_sizes)
            H_cell{group_i} = W{group_i}*S_cell{group_i};
        else
            H_cell{group_i} = X_cell{group_i};
        end
    end
    H = cell2mat(H_cell);
    ata_2I = T'*T+2*eye(m);
    aty1_y2_y3 = T'*Y1-Y2-Y3;
    atx_ate1_j1_p1 = T'*(H-E)+J+P;
    Z = ata_2I\(atx_ate1_j1_p1+ aty1_y2_y3/mu);
    %update E
    xmaz1 = H-T*Z;
    temp1 = xmaz1+Y1/mu;
    E = solve_l1l2(temp1,tau/mu);
    %update P
    P = (Y3+mu*Z)/(lambda*LC+mu*eye(n));
    
    %update Y1-Y3
    leq1 = xmaz1-E;
    leq2 = Z-J;
    leq3 = Z-P;
    stopC1 = max(max(max(max(abs(leq1))),max(max(abs(leq2)))), max(max(abs(leq3))));
%     disp(stopC1);
%     if iter==1 || mod(iter,50)==0 || stopC<tol
%         disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
%             ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
%     end
    if stopC1<tol
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        mu = min(max_mu,mu*rho);
    end
end
end

function [x] = solve_l2(w,tau)
% min tau |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>tau
    x = (nw-tau)*w/nw;
else
    x = zeros(length(w),1);
end
end

function L = computeLaplacianMatrix(M)
[m,n] = size(M);
if m~=n
    L = [];
    return;
end
    
sum_M = sum(M,2);
L = zeros(size(M));
for i = 1:m
    L(i,i) = sum_M(i);
end
L = L-M;
end

function MMD = computeMMD(data1, data2)
% compute MMD between DATA1 and DATA2
% DATA1: DxN1 data samples
% DATA2: DxN1 data samples
% 2018.5.6

diff = sum(data1,2)/size(data1,2)-sum(data2,2)/size(data2,2);
MMD = diff'*diff;
end