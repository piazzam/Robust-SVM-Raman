function [testing_error] = uow_ROBUST_multiclass(DATA,scalar_rho,i_runs,n_runs,users)
% questa function è all'interno del ciclo for:
% 1. partiziona DATA in training set/testing set, attraverso my_tries
% 2. trova la soluzione ottimale sul training set, testando tutte le
% possibili combinazioni di parametri
% 3. testa il separatore ottimale sul testing set
% 4. restituisce il valore del testing error


%%%%%%%%%%%%%%%%%%%%%%%%%%%% partiziono i dati
%fprintf('iterazione %d/%d\n', i_runs, n_runs);
persona = users{i_runs};
DATAtest = DATA(DATA.Var17 == string(persona), :);
% DATAtest = removevars(DATAtest, {'mean_Var1','Var17','GroupCount'});
DATAtest = table2array(DATAtest)';
    
DATAtrain = DATA(DATA.Var17 ~= string(persona), :);
% DATAtrain = removevars(DATAtrain, {'mean_Var1','Var17','GroupCount'});
DATAtrain = table2array(DATAtrain)';

[n,~] = size(DATAtrain);
n = n-1;

m_train = zeros(1,DATAtrain(end,end));  
for j=0:length(m_train)
    m_train(j+1) = length(find(DATAtrain(end,:)==j));
end
m_train_tot = sum(m_train);

m_test = zeros(1,DATAtest(end,end));
for j=0:length(m_test)
    m_test(j+1) = length(find(DATAtest(end,:)==j));
end
m_test_tot = sum(m_test);

L = length(m_train);

dati = DATAtrain(1:end-1,:);
y_label=DATAtrain(end,:)';

% costruisco i dati sulle singole classi (tot 3 classi)
dati_class_1 = DATAtrain(1:end-1,1:m_train(1));
dati_class_2 = DATAtrain(1:end-1,m_train(1)+1:m_train(1)+m_train(2));
dati_class_3 = DATAtrain(1:end-1,m_train(1)+m_train(2)+1:end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% kernel k (matrice K)
K = zeros(m_train_tot,m_train_tot);

% scelta del kernel
% polinomiale di grado d --> k(x,y) = (c+<x,y>)^d
d = 1; c = 0;
%d = 1; c = max(std(dati,0,2));

% RBF con parametro alpha --> k(x,y) = exp(-norm(x-y)^2/(2*alpha^2))
% alpha = max(std(dati,0,2));

% costruisco solo la parte triangolare superiore, poi traspongo
K = untitled(m_train_tot, K, dati, c, d);

% determino il vettore della radice quadrata sulle diagonali
K_d_sqrt = sqrt(diag(K));

%%%%%%%%%%%%%%%%%%%%%%%%%%%% perturbazioni (INPUT space)
% scelta della norma p
p = Inf;

% scegliamo in generale come perturbazione rho*max(std), INDIPENDENTEMENTE
% DA p
std_1 = std(dati_class_1,0,2);
std_2 = std(dati_class_2,0,2);
std_3 = std(dati_class_3,0,2);
rho_1 = scalar_rho;
rho_2 = rho_1;
rho_3 = rho_1;
eta_1 = rho_1*max(std_1);
eta_2 = rho_2*max(std_2);
eta_3 = rho_3*max(std_3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% perturbazioni (FEATURE space)
% costruzione della costante C
if p == inf
    C = sqrt(n);
elseif p <= 2
    C = 1;
elseif p > 2
    C = n^((p-2)/(2*p));
end

% % delta su polinomiale omogenea di grado d
delta_1 = zeros(m_train(1),1);
delta_2 = zeros(m_train(2),1);
delta_3 = zeros(m_train(3),1);
if d == 1
    delta_1 = C*eta_1;
    delta_2 = C*eta_2;
    delta_3 = C*eta_3;
    % costruisco il vettore dei delta
    delta = [delta_1*ones(1,m_train(1)) delta_2*ones(1,m_train(2)) delta_3*ones(1,m_train(3))];
    delta = delta';
end
% if d > 1
%     for i = 1:m_train(1)
%         x_i = dati_class_1(:,i);
%         for k = 1:d
%             delta_1(i) = delta_1(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_1)^k;
%         end
%     end
%     for i = 1:m_train(2)
%         x_i = dati_class_2(:,i);
%         for k = 1:d
%             delta_2(i) = delta_2(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_2)^k;
%         end
%     end
%     for i = 1:m_train(3)
%         x_i = dati_class_3(:,i);
%         for k = 1:d
%             delta_3(i) = delta_3(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_3)^k;
%         end
%     end
%     % costruisco il vettore dei delta
%     delta = [delta_1' delta_2' delta_3'];
%     delta = delta';
% end

% % delta su polinomiale non omogenea di grado d>1 e costante additiva c
% delta_1 = zeros(m_train(1),1);
% delta_2 = zeros(m_train(2),1);
% delta_3 = zeros(m_train(3),1);
% if d > 1
%     % per classe 1
%     aux_j = 0;
%     aux_k = 0;
%     for i = 1:m_train(1)
%         x_i = dati_class_1(:,i);
%         for k = 1:d
%             delta_1(i) = delta_1(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_1)^k;
%         end
%         for k = 1:d-1
%             for j = 1:d-k
%                 aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_1)^j;
%             end
%             aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
%         end
%         delta_1(i) = (delta_1(i))^2 + (aux_k)^2;
%     end
%     delta_1 = sqrt(delta_1);
% 
%     % per classe 2
%     aux_j = 0;
%     aux_k = 0;
%     for i = 1:m_train(2)
%         x_i = dati_class_2(:,i);
%         for k = 1:d
%             delta_2(i) = delta_2(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_2)^k;
%         end
%         for k = 1:d-1
%             for j = 1:d-k
%                 aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_2)^j;
%             end
%             aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
%         end
%         delta_2(i) = (delta_2(i))^2 + (aux_k)^2;
%     end
%     delta_2 = sqrt(delta_2);
% 
%     % per classe 3
%     aux_j = 0;
%     aux_k = 0;
%     for i = 1:m_train(3)
%         x_i = dati_class_3(:,i);
%         for k = 1:d
%             delta_3(i) = delta_3(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_3)^k;
%         end
%         for k = 1:d-1
%             for j = 1:d-k
%                 aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_3)^j;
%             end
%             aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
%         end
%         delta_3(i) = (delta_3(i))^2 + (aux_k)^2;
%     end
%     delta_3 = sqrt(delta_3);
%     % costruisco il vettore dei delta
%     delta = [delta_1' delta_2' delta_3'];
%     delta = delta';
% end



% % delta su RBF Kernel
% delta_1 = sqrt(2-2*exp(-(C*eta_1)^2/(2*alpha^2)));
% delta_2 = sqrt(2-2*exp(-(C*eta_2)^2/(2*alpha^2)));
% delta_3 = sqrt(2-2*exp(-(C*eta_3)^2/(2*alpha^2)));
% delta = [delta_1*ones(1,m_train(1)) delta_2*ones(1,m_train(2)) delta_3*ones(1,m_train(3))];
% delta = delta';

%%%%%%%%%%%%%%%%%%%%%%%%%%%% risolvo ora il pb nominale
%vectornu = logspace(-3,0,5);
vectornu = [0.006003704206442 0.018746888575903 0.069310493609802...
    0.534775938716443 1.433725286800503];

u_vect = zeros(m_train_tot,L);
b_vect = zeros(1,L);
D_hat_tensore = zeros(m_train_tot,m_train_tot,L);

for l=1:L
    y_hat = -ones(m_train_tot,1);
    D_hat = zeros(m_train_tot,m_train_tot);
    y_hat = 1*(y_label==l-1)-1*(y_label~=l-1);
    D_hat = diag(y_hat);

    training_error_opt = Inf;

    for i_nu = 1:length(vectornu)
        nu = vectornu(i_nu);

        cvx_begin quiet
        cvx_solver mosek
        cvx_precision best
        variables u_l(m_train_tot) vargamma_l xi_l(m_train_tot) s_l(m_train_tot)
        minimize sum(s_l) + nu*sum(xi_l)
        subject to
        D_hat*(K*D_hat*u_l-ones(m_train_tot,1)*vargamma_l)+xi_l-delta*(K_d_sqrt'*s_l) >= ones(m_train_tot,1);
        xi_l >= 0;
        u_l >= -s_l;
        u_l <= s_l;
        s_l >= 0;
        cvx_end;
        
        % calcolo omega_1 e omega_2
        omega_minus_l = - min(D_hat*xi_l);
        omega_l = max(D_hat*xi_l);
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%% risolvo la versione di Liu e Potra
        num_points = 1e2;
        discr_b_l = linspace(vargamma_l+1-omega_minus_l,vargamma_l-1+omega_l,num_points);

        max_b = m_train_tot;
        for j = 1:length(discr_b_l)
            b_l = discr_b_l(j);
            if sum((D_hat*(-K*D_hat*u_l+ones(m_train_tot,1)*b_l))>0) < max_b
                max_b = sum(D_hat*(-K*D_hat*u_l+ones(m_train_tot,1)*b_l+delta*(K_d_sqrt'*abs(u_l)))>0);
                b_opt_l = b_l;
            end
        end

        % trovo quanti punti sono misclassified
        tot_num_misclass_training = length(find((D_hat*(-K*D_hat*u_l+ones(m_train_tot,1)*b_opt_l)+delta*(K_d_sqrt'*abs(u_l)))>0));

        training_error = tot_num_misclass_training/m_train_tot; % average

        if training_error < training_error_opt
            training_error_opt = training_error;
            u_opt_l = u_l;
            b_opt_opt_l = b_opt_l;
        end

    end
    u_vect(:,l) = u_opt_l;
    b_vect(:,l) = b_opt_opt_l;
    D_hat_tensore(:,:,l) = D_hat;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% testo il modello sul testing set
miscl_testing = 0;

for j_test = 1:m_test_tot
    x_test = DATAtest(1:n,j_test);
    y_test = DATAtest(end,j_test);
    K_test = zeros(m_train_tot,1);
    for j = 1:m_train_tot
        K_test(j) = (dati(:,j)'*x_test+c)^d;
        %K_test(j) = exp(-norm(dati(:,j)-x_test)^2/(2*alpha^2));
    end
    fun_val = zeros(1,L);
    for l = 1:L
        fun_val(l)= K_test'*D_hat_tensore(:,:,l)*u_vect(:,l)-b_vect(:,l);
    end
    class_l_star = find(fun_val==max(fun_val));
    if class_l_star ~= y_test + 1
        miscl_testing = miscl_testing+1;
    end
end

testing_error = miscl_testing/m_test_tot;
end


function K = untitled(m_train_tot, K, dati, c, d)
i = 1; j = 1;
while (i<=m_train_tot)
    while (j<=m_train_tot)
        K(i,j) = (dati(:,i)'*dati(:,j)+c)^d;
        %K(i,j) = exp(-norm(dati(:,i)-dati(:,j))^2/(2*alpha^2));
        j = j+1;
    end
    i = i+1;
    j = i;
end
K = K+K';
for i = 1:m_train_tot
    K(i,i) = K(i,i)/2;
end
end