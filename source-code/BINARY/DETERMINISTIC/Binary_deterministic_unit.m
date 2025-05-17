function [TP, FN, FP, TN, testing_error, best_nu] = Binary_deterministic_unit(DATA, persona,lap)
    % questa function è all'interno del ciclo for reso parallelo: partiziona i
    % dati, trova la soluzione sul training set, testa il testing set,
    % restituisce il testing error
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% partiziono i dati
    
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
    c_vect = {0,0,0,max(std(dati,0,2)),max(std(dati,0,2)),max(std(dati,0,2))};
    d_vect = {1,2,3,1,2,3};
    c = c_vect{lap};
    d = d_vect{lap};
    
    % RBF con parametro alpha --> k(x,y) = exp(-norm(x-y)^2/(2*alpha^2))
    alpha = max(std(dati,0,2));
    
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
        D*(K*D*u-ones(m,1)*vargamma)+xi >= ones(m,1);
        xi >= 0;
        u >= -s;
        u <= s;
        s >= 0;
        cvx_end;
        %disp('  cvx fatto')
    
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
    
        training_error = tot_num_misclass_training/m; % average
    
        if training_error < training_error_opt
            training_error_opt = training_error;
            u_opt = u;
            b_opt_opt = b_opt;
            selected_nu = nu;
        end
    end
    best_nu = selected_nu;
    
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
end