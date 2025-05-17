format long
clear
close all
clc

%DATA = readtable("15_componenti.csv");
%DATA = DATA(2:end,:);
%DATA = DATA(DATA.Var17 ~= 0, :);

DATA = readtable("new_dataset/new_experiments/ten_windows/avg/interval_9.csv");
DATA = removevars(DATA, "Var111");

%users = unique(DATA.Var16);

users = readtable("new_dataset/automatic_cleaned/avg/user.csv", 'Delimiter', ';', 'ReadVariableNames', false);
labels = readtable("new_dataset/automatic_cleaned/avg/labels.csv",  'Delimiter', ';', 'ReadVariableNames', false);

DATA = addvars(DATA, users.Var1);
DATA = addvars(DATA, labels.Var1);

label_a = 0;
label_b = 2;
labels_excluded = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%% testo il tutto per n_runs volte
n_runs = length(users.Var1);

vect_TP = zeros(n_runs,1);
vect_FN = zeros(n_runs,1);
vect_FP = zeros(n_runs,1);
vect_TN = zeros(n_runs,1);
vect_testing_error = zeros(n_runs,1);
vect_best_nu = zeros(n_runs,1);
vect_best_rho = zeros(n_runs,1);

tic

res = zeros(101, 4);

for i_runs = 1:numel(users.Var1)
    if labels.Var1(i_runs) == labels_excluded
        continue;
    end
    fprintf('    iterazione %d/%d\n', i_runs, n_runs);
    persona = users.Var1{i_runs};
    
    [Atrain, Atest, Btrain, Btest] = my_tries(DATA,persona, label_a, label_b);

    dati_set_A = Atrain';
    dati_set_B = Btrain';
    
    [~,m_A] = size(dati_set_A);
    [~,m_B] = size(dati_set_B);
    
    dati = [dati_set_A dati_set_B];
    [n,m] = size(dati); % dimensione e numero dei dati

    y = [ones(1,m_A) -ones(1,m_B)];
    D = diag(y); % matrice con le label

    %%%%%%%%%%%%%%%%%%%%%%%%%%%% kernel k (matrice K)
    K = zeros(m,m);
    
    % scelta del kernel
    % polinomiale di grado d --> k(x,y) = (<x,y>+c)^d
    c = 0;
    % c = max(std(dati,0,2);
    d = 3;
    % RBF con parametro alpha --> k(x,y) = exp(-norm(x-y)^2/(2*alpha^2))
    %alpha_k = max(std(dati,0,2));
    
    % costruisco solo la parte triangolare superiore, poi traspongo
    if exist('alpha_k', 'var')
        D = pdist(dati', 'euclidean');
        D = squareform(D);
        K = exp(-D.^2 / (2 * alpha_k^2));
        K = K + K';
        K = K/2;
    else
         K = (dati' * dati + c) .^ d;
    end

    K_d_sqrt = sqrt(diag(K));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%% perturbazioni (INPUT space)
    % scelta della norma p
    p = inf;
    
    % scegliamo in generale come perturbazione rho*max(std), INDIPENDENTEMENTE
    std_A = std(dati_set_A,0,2);
    std_B = std(dati_set_B,0,2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%% perturbazioni (FEATURE space)
    % costruzione della costante C
    if p == inf
        C = sqrt(n);
    elseif p <= 2
        C = 1;
    elseif p > 2
        C = n^((p-2)/(2*p));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%% risolvo ora il pb nominale
    % Definisci lo spazio di ricerca per nu
    nu_range = [0, 2];  % Intervallo di valori di nu
    var1 = optimizableVariable('nu', nu_range, 'Type', 'real');
    rho_range = [10^(-7), 10^(-1)];
    var2 = optimizableVariable('scalar_rho', rho_range, 'Type', 'real');
    vars = [var1,var2];

    % Definisci la funzione obiettivo per l'ottimizzazione bayesiana
    objective = @(vars) calculate_error(vars.nu, D, K, m, std_A, std_B, m_A, m_B, C, ...
        dati_set_A, dati_set_B, vars.scalar_rho, d, K_d_sqrt, c); %,alpha_k);

    % Configura l'ottimizzazione bayesiana
    results = bayesopt(objective, vars, UseParallel=false, Verbose=0, PlotFcn=[], MaxObjectiveEvaluations=1);
    %results = bayesopt(objective, vars, AcquisitionFunctionName="probability-of-improvement", UseParallel=false, Verbose=1, PlotFcn=[]);
    
    % Ottieni il miglior valore di nu
    best_nu = results.XAtMinObjective.nu;
    best_scalar_rho = results.XAtMinObjective.scalar_rho;
    vect_best_nu(i_runs) = best_nu;
    vect_best_rho(i_runs) = best_scalar_rho;

    rho_A = best_scalar_rho;
    rho_B = rho_A;
    eta_A = rho_A*max(std_A);
    eta_B = rho_B*max(std_B);    
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
    % if d > 1 
    %     % per classe A
    %     aux_j = 0;
    %     aux_k = 0;
    %     for i = 1:m_A
    %         x_i = dati_set_A(:,i);
    %         for k = 1:d
    %             delta_A(i) = delta_A(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_A)^k;
    %         end
    %         for k = 1:d-1
    %             for j = 1:d-k
    %                 aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_A)^j;
    %             end
    %             aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
    %         end
    %         delta_A(i) = (delta_A(i))^2 + (aux_k)^2;
    %     end
    %     delta_A = sqrt(delta_A);
    % 
    %     % per classe B
    %     aux_j = 0;
    %     aux_k = 0;
    %     for i = 1:m_B
    %         x_i = dati_set_B(:,i);
    %         for k = 1:d
    %             delta_B(i) = delta_B(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_B)^k;
    %         end
    %         for k = 1:d-1
    %             for j = 1:d-k
    %                 aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_B)^j;
    %             end
    %             aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
    %         end
    %         delta_B(i) = (delta_B(i))^2 + (aux_k)^2;
    %     end
    %     delta_B = sqrt(delta_B);
    %     % costruisco il vettore dei delta
    %     delta = [delta_A' delta_B'];
    %     delta = delta';
    % end
     
    % delta su RBF Kernel
    %delta_A = sqrt(2-2*exp(-(C*eta_A)^2/(2*alpha_k^2)));
    %delta_B = sqrt(2-2*exp(-(C*eta_B)^2/(2*alpha_k^2)));
    %delta = [delta_A*ones(1,m_A) delta_B*ones(1,m_B)];
    %delta = delta';

    cvx_begin quiet
    cvx_solver mosek
    cvx_precision high
    variables u(m) vargamma xi(m) s(m)
    minimize sum(s) + best_nu*sum(xi)
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
    
    b_opt_opt = b_opt;
    u_opt = u;

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
            if exist('alpha_k', 'var')
            K_test(j) = exp(-norm(dati(:,j)-xtest)^2/(2*alpha_k^2));
            else
                 K_test(j) = (dati(:,j)'*xtest+c)^d;
            end
        end
        if (K_test'*D*u_opt-b_opt_opt > 0)
            truepositive = truepositive + 1;
            res(i_runs,1) = res(i_runs,1)+1;
        else
            falsenegative = falsenegative + 1;
            res(i_runs,2) = res(i_runs,2)+1;
        end
    end
    
    for i = 1:m_Btest
        K_test = zeros(m,1);
        xtest = Btest(:,i);
        for j = 1:m
            if exist('alpha_k', 'var')
                K_test(j) = exp(-norm(dati(:,j)-xtest)^2/(2*alpha_k^2));
            else
                 K_test(j) = (dati(:,j)'*xtest+c)^d;
            end
        end
        if (K_test'*D*u_opt-b_opt_opt <= 0)
            truenegative = truenegative + 1;
            res(i_runs,3) = res(i_runs,3)+1;
        else
            falsepositive = falsepositive + 1;
            res(i_runs,4) = res(i_runs,4)+1;

        end
    end
    
    TP = truepositive;
    FN = falsenegative;
    FP = falsepositive;
    TN = truenegative;
    
    tot_num_misclass_testing = falsenegative + falsepositive;
    testing_error = tot_num_misclass_testing/(m_Atest+m_Btest); % average

    vect_testing_error(i_runs) = testing_error;
    vect_best_nu(i_runs) = best_nu;
    vect_best_rho(i_runs) = best_scalar_rho;
    vect_TP(i_runs) = TP;
    vect_FN(i_runs) = FN;
    vect_FP(i_runs) = FP;
    vect_TN(i_runs) = TN;
end

toc

mean_all = mean(vect_testing_error);
fprintf('mean testing accuracy %.2f\n', (1-mean_all)*100);
std_all = std(vect_testing_error);
fprintf('std testing error %.2f\n', std_all*100);
precision = sum(vect_TP)/(sum(vect_TP)+sum(vect_FP));
fprintf('precision %.2f\n', precision*100);
sensitivity = sum(vect_TP)/(sum(vect_TP)+sum(vect_TN));
fprintf('sensitivity %.2f\n', sensitivity*100);
specificity = sum(vect_TN)/(sum(vect_TP)+sum(vect_TN));
fprintf('specificity %.2f\n', specificity*100);
MCC = ((sum(vect_TP)*sum(vect_TN))-(sum(vect_FP)*sum(vect_FN)))/sqrt((sum(vect_TP)+sum(vect_FP))*(sum(vect_TP)+sum(vect_FN))*(sum(vect_TN)+sum(vect_FP))*(sum(vect_TN)+sum(vect_FN)));
fprintf('MCC %.2f\n', MCC);
    
disp('TP')
disp(vect_TP)
disp('TN')
disp(vect_TN)
disp('FP')
disp(vect_FP)
disp('FN')
disp(vect_FN)
disp('best_nu')
disp(vect_best_nu)
disp('vect_testing_error')
disp(vect_testing_error)
disp('vect_best_rho')
disp(vect_best_rho)


function error = calculate_error(nu, D, K, m, std_A, std_B, m_A, m_B, C, dati_set_A, dati_set_B, scalar_rho, d, K_d_sqrt, c, alpha_k)   
    
    rho_A = scalar_rho;
    rho_B = rho_A;
    eta_A = rho_A*max(std_A);
    eta_B = rho_B*max(std_B);    
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
    % 
    % % delta su polinomiale non omogenea di grado d>1 e costante additiva c
    % %delta_A = zeros(m_A,1);
    % %delta_B = zeros(m_B,1);
    % if d > 1
    %     % per classe A
    %     aux_j = 0;
    %     aux_k = 0;
    %     for i = 1:m_A
    %         x_i = dati_set_A(:,i);
    %         for k = 1:d
    %             delta_A(i) = delta_A(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_A)^k;
    %         end
    %         for k = 1:d-1
    %             for j = 1:d-k
    %                 aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_A)^j;
    %             end
    %             aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
    %         end
    %         delta_A(i) = (delta_A(i))^2 + (aux_k)^2;
    %     end
    %     delta_A = sqrt(delta_A);
    % 
    %     % per classe B
    %     aux_j = 0;
    %     aux_k = 0;
    %     for i = 1:m_B
    %         x_i = dati_set_B(:,i);
    %         for k = 1:d
    %             delta_B(i) = delta_B(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_B)^k;
    %         end
    %         for k = 1:d-1
    %             for j = 1:d-k
    %                 aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_B)^j;
    %             end
    %             aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
    %         end
    %         delta_B(i) = (delta_B(i))^2 + (aux_k)^2;
    %     end
    %     delta_B = sqrt(delta_B);
    %     % costruisco il vettore dei delta
    %     delta = [delta_A' delta_B'];
    %     delta = delta';
    % end
     
    % delta su RBF Kernel
    % delta_A = sqrt(2-2*exp(-(C*eta_A)^2/(2*alpha_k^2)));
    % delta_B = sqrt(2-2*exp(-(C*eta_B)^2/(2*alpha_k^2)));
    % delta = [delta_A*ones(1,m_A) delta_B*ones(1,m_B)];
    % delta = delta';
    % fprintf('questo è delta_A %.5f\n', delta_A);
    % fprintf('questo è delta_B %.5f\n', delta_B);
    % disp(delta)

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
    error = tot_num_misclass_training/m;
end