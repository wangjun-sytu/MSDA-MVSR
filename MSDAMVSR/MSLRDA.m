function [W,Z,E,iter] = MSLRDA(S, T, S_group_sizes, tau, lambda, LC, Z_init, E_init)
%multiple sources Low-rank Domain adaptation
% inputs:
%        S -- D*N data matrix, D is the data dimension, and N is total number
%             of data vectors.
%        T -- D*M matrix of a dictionary, M is the size of the dictionary
%        S_group_sizes -- a vector recording the size of each group in S.
%        e.g. S_group_sizes = [8,8,10], N should be 8+8+10

tol = 1e-8;
maxIter = 500;
d = size(S,1);
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

function [E] = solve_l1l2(W,tau)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),tau);
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