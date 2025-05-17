format long
clear
close all
clc

DATA = readtable("15_componenti.csv");
DATA = DATA(2:end,:);
DATA = DATA(DATA.Var17 ~= 0, :);

users = unique(DATA.Var16);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% testo il tutto per n_runs volte
n_runs = length(users);

tic

vect_TP = zeros(n_runs,1);
vect_FN = zeros(n_runs,1);
vect_FP = zeros(n_runs,1);
vect_TN = zeros(n_runs,1);
vect_testing_error = zeros(n_runs,1);
vect_best_nu = zeros(n_runs,1);

for i_runs = 1:numel(users)
    fprintf('    iterazione %d/%d\n', i_runs, n_runs);
    persona = users{i_runs};
    
    [Atrain, Atest, Btrain, Btest] = my_tries(DATA,persona);

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
    d = 1;
    % RBF con parametro alpha --> k(x,y) = exp(-norm(x-y)^2/(2*alpha^2))
    %alpha = max(std(dati,0,2));
    
    % costruisco solo la parte triangolare superiore, poi traspongo
    if exist('alpha', 'var')
        D = pdist(dati', 'euclidean');
        D = squareform(D);
        K = exp(-D.^2 / (2 * alpha^2));
        K = K + K';
        K = K/2;
    else
        K = (dati' * dati + c) .^ d;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% risolvo ora il pb nominale
    % Definisci la funzione obiettivo per l'ottimizzazione bayesiana
    objective = @(nu) calculate_error(nu, D, K, m);

    % Definisci lo spazio di ricerca per nu
    nu_range = [0, 2];  % Intervallo di valori di nu
    vars = optimizableVariable('nu', nu_range, 'Type', 'real');

    % Configura l'ottimizzazione bayesiana
    results = bayesopt(objective, vars, UseParallel=false, Verbose=0, PlotFcn="all", MaxObjectiveEvaluations=30);
    % results = bayesopt(objective, vars, AcquisitionFunctionName="probability-of-improvement", UseParallel=false, Verbose=1, PlotFcn=[]);
    
    % Ottieni il miglior valore di nu
    best_nu = results.XAtMinObjective.nu;
    vect_best_nu(i_runs) = best_nu;

    %calcola valori con il nu migliore trovato
    cvx_begin quiet
    cvx_solver mosek
    cvx_precision high
    variables u(m) vargamma xi(m) s(m)
    minimize sum(s) + best_nu*sum(xi)
    subject to
    D*(K*D*u-ones(m,1)*vargamma)+xi >= ones(m,1);
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

    b_opt = vargamma;
    max_b = m;
    prova1 = -K*D*u;
    for j = 1:length(discr_b)
        b = discr_b(j);
        prova2 = D*(prova1+ones(m,1)*b);
        if sum(prova2>0) < max_b
            max_b = sum(prova2>0);
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
            if exist('alpha', 'var')
                K_test(j) = exp(-norm(dati(:,j)-xtest)^2/(2*alpha^2));
            else
                K_test(j) = (dati(:,j)'*xtest+c)^d;
            end
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
            if exist('alpha', 'var')
                K_test(j) = exp(-norm(dati(:,j)-xtest)^2/(2*alpha^2));
            else
                K_test(j) = (dati(:,j)'*xtest+c)^d;
            end
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

    vect_testing_error(i_runs) = testing_error;
    vect_best_nu(i_runs) = best_nu;
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

function error = calculate_error(nu, D, K, m)
    cvx_begin quiet
    cvx_solver mosek
    cvx_precision high
    variables u(m) vargamma xi(m) s(m)
    minimize sum(s) + nu{1,1}*sum(xi)
    subject to
    D*(K*D*u-ones(m,1)*vargamma)+xi >= ones(m,1);
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
    
    b_opt = vargamma;
    max_b = m;
    prova1 = -K*D*u;
    for j = 1:length(discr_b)
        b = discr_b(j);
        prova2 = D*(prova1+ones(m,1)*b);
        if sum(prova2>0) < max_b
            max_b = sum(prova2>0);
            b_opt = b;
        end
    end
    % trovo quanti punti sono misclassified
    tot_num_misclass_training = length(find(D*(-K*D*u+ones(m,1)*b_opt)>0));   
    error = tot_num_misclass_training/m;
end