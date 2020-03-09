function [A, objs] = MVSRC(q, data, view_sizes, eta, phi, max_iter)
% q: query sample, Dx1;
% data: samples in dictionary, DxN
% A: N*M
% data_sizes: the data numbers in each task
% view_sizes: the feature numbers in each view
% Jun Wang UNC-CH
% 2017.8.6

A_init = rand(size(data,2), length(view_sizes));
A = A_init;

for iter = 1:max_iter
    [A_new{iter}, objs(iter)] = stepMVSRC(q, data, view_sizes, A, eta, phi, iter);
    A = A_new{iter};
end
end

function [A_new, obj] = stepMVSRC(q, data, view_sizes, A, eta, phi, iter)
% A: N*M

% update A
f_der_at_A =  getDerivative_f_at_A(data, q, A, view_sizes, phi);

step_size_A = 1e-10; %Armijo_A(q, data, task_sizes, view_sizes, A, Theta, eta, phi, f_der_at_A);

A_new = proximalOperator(A-step_size_A*f_der_at_A, 'NUCLEAR', 2*step_size_A, eta);

% %%%%%%
obj = objective(q, data,view_sizes, A_new, eta, phi);
% fprintf('(%d)%f\n',iter, obj);
% %%%%%%%
end

function obj  = objective(q, data, view_sizes, A, eta, phi)
view_n = length(view_sizes);
loss = 0;
for view_i = 1:view_n
    ind1 = sum(view_sizes(1:view_i-1))+1;
    ind2 = sum(view_sizes(1:view_i));
    q_viewi = q(ind1:ind2,:);
    data_viewi = data(ind1:ind2,:);
    alpha_viewi = A(:,view_i);
    diff_viewi = q_viewi-data_viewi*alpha_viewi;
    loss = loss+diff_viewi'*diff_viewi;
end

[U,S,V] = svd(A,'econ');
A_nuclear = sum(diag(abs(S)));

A_L21 = 0;
for i = 1:size(A,1)
    A_L21 = A_L21+norm(A(i,:),2);
end

obj = loss+eta*A_nuclear+phi*A_L21;
end

function f_der_A = getDerivative_f_at_A(data, q, A, view_sizes, phi)
data_n = size(data,2);
view_n = length(view_sizes);
zeros(data_n,view_n);
f1_der = zeros(data_n, view_n);
for view_i = 1:view_n
    ind1 = sum(view_sizes(1:view_i-1))+1;
    ind2 = sum(view_sizes(1:view_i));
    q_viewi = q(ind1:ind2,:);
    data_viewi = data(ind1:ind2,:);
    alpha_viewi = A(:,view_i);
    f1_der(:,view_i) = 2*(-data_viewi'*q_viewi+data_viewi'*data_viewi*alpha_viewi);
end

Lambda = zeros(size(A,1));
for i = 1:size(A,1)
    Lambda(i,i) = 0.5/norm(A(i,:),2);
end
f2_der = 2*Lambda*A;
f_der_A = f1_der+phi*f2_der;
end

function Z = proximalOperator(X, type, rho, lambda)
switch type
    case 'NUCLEAR'
        % argmin{0.5*rho*||Z-X||^2+lambda*||Z||*}
        [U,S,V] = svd(X,'econ');
        Z = U*diag(pos(diag(S)-lambda/rho))*V';
        if (Z==0)
            Z = X;
        end
    case 'L21'
        % argmin{0.5*rho*||Z-X||^2+lambda*||Z||_21}
        X_norm = sqrt(sum(X.*X,2));
        diff = (X_norm-lambda/rho)./X_norm;
        neg_ind = diff<0;
        diff(neg_ind) = 0;
        Z = X.*repmat(diff,1,size(X,2));
    otherwise
        ;
end
end

function pos_x = pos(x)
pos_ind = (x>0);
pos_x = x;
pos_x(~pos_ind) = 0;
end