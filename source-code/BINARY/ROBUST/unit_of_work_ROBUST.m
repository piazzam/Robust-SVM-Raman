function [TP, FN, FP, TN, testing_error, testing_error_A, testing_error_B] = unit_of_work_ROBUST(DATA,scalar_rho,i_runs,n_runs,users,persona)
% function [TP, FN, FP, TN, testing_error, testing_error_A, testing_error_B] = unit_of_work_ROBUST(DATA)
%function [TP, FN, FP, TN, testing_error] = unit_of_work_ROBUST(DATA)
% questa function è all'interno del ciclo for reso parallelo: partiziona i
% dati, trova la soluzione sul training set, testa il testing set,
% restituisce il testing error

%%%%%%%%%%%%%%%%%%%%%%%%%%%% partiziono i dati
[Atrain, Atest, Btrain, Btest] = my_tries(DATA, persona);

dati_set_A = Atrain';
dati_set_B = Btrain';

[~,m_A] = size(dati_set_A);
[~,m_B] = size(dati_set_B);

dati = [dati_set_A dati_set_B];
[n,m] = size(dati); % dimensione e numero dei dati

dati_set_A = dati(:,1:m_A);
dati_set_B = dati(:,m_A+1:end);
y = [ones(1,m_A) -ones(1,m_B)];
D = diag(y); % matrice con le label


%%%%%%%%%%%%%%%%%%%%%%%%%%%% kernel k (matrice K)
K = zeros(m,m);

% scelta del kernel
% polinomiale di grado d --> k(x,y) = (<x,y>+c)^d
d = 1; c = 0;
%d = 2; c = max(std(dati,0,2)); %%% ricorda di cambiare sotto!!!

% RBF con parametro alpha --> k(x,y) = exp(-norm(x-y)^2/(2*alpha^2))
%alpha = 2;
%alpha = max(std(dati,0,2));

% costruisco solo la parte triangolare superiore, poi traspongo
% i = 1; j = 1;
% while (i<=m)
%     while (j<=m)
%         K(i,j) = (dati(:,i)'*dati(:,j)+c)^d;
%         %K(i,j) = exp(-norm(dati(:,i)-dati(:,j))^2/(2*alpha^2));
%         j = j+1;
%     end
%     i = i+1;
%     j = i;
% end
% K = K+K';
% for i = 1:m
%     K(i,i) = K(i,i)/2;
% end
K = (dati' * dati + c) .^ d;
% determino max sulla diagonale
% K_d_max = max(diag(K));

% determino il vettore della radice quadrata sulle diagonali
K_d_sqrt = sqrt(diag(K));


%%%%%%%%%%%%%%%%%%%%%%%%%%%% perturbazioni (INPUT space)
% scelta della norma p
p = inf;

% scegliamo in generale come perturbazione rho*max(std), INDIPENDENTEMENTE
% DA p
std_A = std(dati_set_A,0,2);
std_B = std(dati_set_B,0,2);
rho_A = scalar_rho;
rho_B = rho_A;
eta_A = rho_A*max(std_A);
eta_B = rho_B*max(std_B);
%eta_A = rho_A;
%eta_B = rho_B;


% % costruisco il vettore degli eta
% eta = [eta_A*ones(1,m_A) eta_B*ones(1,m_B)];
% eta = eta';

%%%%%%%%%%%%%%%%%%%%%%%%%%%% perturbazioni (FEATURE space)
% costruzione della costante C
if p == inf
    C = sqrt(n);
elseif p <= 2
    C = 1;
elseif p > 2
    C = n^((p-2)/(2*p));
end

% % % delta su polinomiale omogenea di grado d
delta_A = zeros(m_A,1);
delta_B = zeros(m_B,1);
if d == 1
     delta_A = C*eta_A;
     delta_B = C*eta_B;
     % costruisco il vettore dei delta
     delta = [delta_A*ones(1,m_A) delta_B*ones(1,m_B)];
     delta = delta';
 end
 if d > 1
     for i = 1:m_A
         x_i = dati_set_A(:,i);
         for k = 1:d
             delta_A(i) = delta_A(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_A)^k;
         end
     end
     for i = 1:m_B
         x_i = dati_set_B(:,i);
         for k = 1:d
             delta_B(i) = delta_B(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_B)^k;
         end
     end
     % costruisco il vettore dei delta
     delta = [delta_A' delta_B'];
     delta = delta';
 end

% delta su polinomiale non omogenea di grado d>1 e costante additiva c
%delta_A = zeros(m_A,1);
%delta_B = zeros(m_B,1);
if d > 1
    % per classe A
    aux_j = 0;
    aux_k = 0;
    for i = 1:m_A
        x_i = dati_set_A(:,i);
        for k = 1:d
            delta_A(i) = delta_A(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_A)^k;
        end
        for k = 1:d-1
            for j = 1:d-k
                aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_A)^j;
            end
            aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
        end
        delta_A(i) = (delta_A(i))^2 + (aux_k)^2;
    end
    delta_A = sqrt(delta_A);

    % per classe B
    aux_j = 0;
    aux_k = 0;
    for i = 1:m_B
        x_i = dati_set_B(:,i);
        for k = 1:d
            delta_B(i) = delta_B(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_B)^k;
        end
        for k = 1:d-1
            for j = 1:d-k
                aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_B)^j;
            end
            aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
        end
        delta_B(i) = (delta_B(i))^2 + (aux_k)^2;
    end
    delta_B = sqrt(delta_B);
    % costruisco il vettore dei delta
    delta = [delta_A' delta_B'];
    delta = delta';
end



% % delta su RBF Kernel
% delta_A = sqrt(2-2*exp(-(C*eta_A)^2/(2*alpha^2)));
% delta_B = sqrt(2-2*exp(-(C*eta_B)^2/(2*alpha^2)));
% delta = [delta_A*ones(1,m_A) delta_B*ones(1,m_B)];
% delta = delta';

%%%%%%%%%%%%%%%%%%%%%%%%%%%% risolvo ora il pb robusto
%vectornu = logspace(-3,0,5);
vectornu = [0.006003704206442 0.018746888575903 0.069310493609802...
    0.534775938716443 1.433725286800503];

training_error_opt = Inf;

for i_nu = 1:length(vectornu)
    nu = vectornu(i_nu);

    cvx_begin quiet
    cvx_solver mosek
    cvx_precision high
    variables u(m) vargamma xi(m) s(m)
    minimize sum(s) + nu*sum(xi)
    subject to
    % D*(K*D*u-ones(m,1)*vargamma)+xi-delta*sqrt(K_d_max)*sum(s) >= ones(m,1);
    D*(K*D*u-ones(m,1)*vargamma)+xi-delta*(K_d_sqrt'*s) >= ones(m,1);
    xi >= 0;
    u >= -s;
    u <= s;
    s >= 0;
    cvx_end;

    % calcolo omega_1 e omega_2
    omega_A = max(D*xi);
    omega_B = max(-D*xi);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%% risolvo la versione di Liu e Potra
    num_points = 1e2;
    discr_b = linspace(vargamma+1-omega_B,vargamma-1+omega_A,num_points);

    max_b = m;
    b_opt = vargamma;
    for j = 1:length(discr_b)
        b = discr_b(j);
        if sum((D*(-K*D*u+ones(m,1)*b)+delta*(K_d_sqrt'*abs(u)))>0) < max_b
            max_b = sum((D*(-K*D*u+ones(m,1)*b)+delta*(K_d_sqrt'*abs(u)))>0);
            b_opt = b;
        end
    end

    % trovo quanti punti sono misclassified
    tot_num_misclass_training = length(find((D*(-K*D*u+ones(m,1)*b_opt)+delta*(K_d_sqrt'*abs(u)))>0));

    training_error = tot_num_misclass_training/m; % average

    if training_error < training_error_opt
        training_error_opt = training_error;
        u_opt = u;
        b_opt_opt = b_opt;
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% testo il modello sul testing set
Atest = Atest';
Btest = Btest';

[~,m_Atest] = size(Atest);
[~,m_Btest] = size(Btest);

truepositive = 0;
falsepositive = 0;
truenegative = 0;
falsenegative = 0;


for i = 1:m_Atest
    K_test = zeros(m,1);
    xtest = Atest(:,i);
    for j = 1:m
        K_test(j) = (dati(:,j)'*xtest+c)^d;
        %K_test(j) = exp(-norm(dati(:,j)-xtest)^2/(2*alpha^2));
    end
    if (K_test'*D*u_opt-b_opt_opt > 0)
        truepositive = truepositive + 1;
    else
        falsenegative = falsenegative + 1;
    end
end

for i = 1:m_Btest
    K_test = zeros(m,1);
    xtest = Btest(:,i);
    for j = 1:m
        K_test(j) = (dati(:,j)'*xtest+c)^d;
        %K_test(j) = exp(-norm(dati(:,j)-xtest)^2/(2*alpha^2));
    end
    if (K_test'*D*u_opt-b_opt_opt <= 0)
        truenegative = truenegative + 1;
    else
        falsepositive = falsepositive + 1;
    end
end

TP = truepositive;
FN = falsenegative;
FP = falsepositive;
TN = truenegative;

tot_num_misclass_testing = falsenegative + falsepositive;

testing_error = tot_num_misclass_testing/(m_Atest+m_Btest); % average

testing_error_A = falsenegative/m_Atest;
testing_error_B = falsepositive/m_Btest;

end