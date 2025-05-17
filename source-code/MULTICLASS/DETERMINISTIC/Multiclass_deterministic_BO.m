% Main per il problema MULTICLASS deterministic
format long
clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%% importo i dati
DATA = readtable("15_componenti.csv");
DATA = DATA(2:end,:);

users = unique(DATA.Var16);
%%%%%%%%%%%%%%%%%%%%%%%%%%%% testo il tutto per il numero di users distinci
n_runs = numel(users);

for lap = 1:1
    fprintf('lap %d/%d\n', lap, 6);
    vect_TP0 = zeros(length(users),1);
    vect_TP1 = zeros(length(users),1);
    vect_TP2 = zeros(length(users),1);
    vect_F01 = zeros(length(users),1);
    vect_F02 = zeros(length(users),1);
    vect_F10 = zeros(length(users),1);
    vect_F12 = zeros(length(users),1);
    vect_F20 = zeros(length(users),1);
    vect_F21 = zeros(length(users),1);
    vect_testing_error = zeros(length(users),1);
    vect_sensitivity = zeros(n_runs,1);
    vect_specificity = zeros(n_runs,1);
    vect_best_nu = zeros(n_runs,3);
    % vect_MCC = zeros(n_runs,1);
    % vect_ER = zeros(n_runs,1);    
    
    % risolvo il modello
    tic
    for i_runs = 1:numel(users)
    %for p = 1:1
        fprintf('  iterazione %d/%d\n', i_runs, numel(users));
        persona = users{i_runs};
        DATAtest = DATA(DATA.Var16 == string(persona), :);
        DATAtest = removevars(DATAtest, {'Var16'});
        DATAtest = table2array(DATAtest)';
    
        DATAtrain = DATA(DATA.Var16 ~= string(persona), :);
        DATAtrain = removevars(DATAtrain, {'Var16'});
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


        %%%%%%%%%%%%%%%%%%%%%%%%%%%% kernel k (matrice K)
        K = zeros(m_train_tot,m_train_tot);

        % scelta del kernel
        % polinomiale di grado d --> k(x,y) = (c+<x,y>)^d
        c_vect = {0,0,0,max(std(dati,0,2)),max(std(dati,0,2)),max(std(dati,0,2))};
        d_vect = {1,2,3,1,2,3};
        c = c_vect{lap};
        d = d_vect{lap};

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
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%% risolvo ora il modello deterministico 
        u_vect = zeros(m_train_tot,L);
        b_vect = zeros(1,L);
        D_hat_tensore = zeros(m_train_tot,m_train_tot,L);

        for l=1:L
            %fprintf('  %d/%d\n', l, L);
            y_hat = -ones(m_train_tot,1);
            D_hat = zeros(m_train_tot,m_train_tot);
            y_hat = 1*(y_label==l-1)-1*(y_label~=l-1);
            D_hat = diag(y_hat);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%% risolvo ora il pb nominale
            % Definisci la funzione obiettivo per l'ottimizzazione bayesiana
            objective = @(nu) calculate_error(nu, D_hat, K, m_train_tot);

            % Definisci lo spazio di ricerca per nu
            nu_range = [0, 2]; % Intervallo di valori di nu
            vars = optimizableVariable('nu', nu_range, 'Type', 'real');

            % Configura l'ottimizzazione bayesiana
            results = bayesopt(objective, vars, UseParallel=false, Verbose=1, PlotFcn=[], MaxObjectiveEvaluations=30);
            %results = bayesopt(objective, vars, AcquisitionFunctionName="probability-of-improvement", UseParallel=false, Verbose=1, PlotFcn=[], MaxObjectiveEvaluations=30);
    
            % Ottieni il miglior valore di nu
            best_nu = results.XAtMinObjective.nu;
            vect_best_nu(i_runs,l) = best_nu;
            
            cvx_begin quiet
            cvx_solver mosek
            cvx_precision high
            variables u_l(m_train_tot) vargamma_l xi_l(m_train_tot) s_l(m_train_tot)
            minimize sum(s_l) + best_nu*sum(xi_l)
            subject to
            D_hat*(K*D_hat*u_l-ones(m_train_tot,1)*vargamma_l)+xi_l >= ones(m_train_tot,1);
            xi_l >= 0;
            u_l >= -s_l;
            u_l <= s_l;
            s_l >= 0;
            cvx_end;
            
            % calcolo i valori di omega
            omega_minus_l = - min(D_hat*xi_l);
            omega_l = max(D_hat*xi_l);
        
            %%%%%%%%%%%%%%%%%%%%%%%%%%%% risolvo la versione non lineare di Liu e Potra
            num_points = 1e2;
            discr_b_l = linspace(vargamma_l+1-omega_minus_l,vargamma_l-1+omega_l,num_points);
        
            max_b = m_train_tot;
            prova1 = -K*D_hat*u_l;
            for j = 1:length(discr_b_l)
                b_l = discr_b_l(j);
                prova2 = D_hat*(prova1+ones(m_train_tot,1)*b_l);
                if sum(prova2>0) < max_b
                    max_b = sum(prova2>0);
                    b_opt_l = b_l;
                end
            end
            
            u_vect(:,l) = u_l;
            b_vect(:,l) = b_opt_l;
            D_hat_tensore(:,:,l) = D_hat;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%% testo il modello sul testing set
        miscl_testing = 0;
        predc = zeros(m_test_tot,1);
        label = zeros(m_test_tot,1);

        for j_test = 1:m_test_tot
            %fprintf('      %d/%d\n',j_test, m_test_tot)
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
            label(j_test) = y_test;
            predc(j_test) = class_l_star - 1;        
        end

        testing_error = miscl_testing/m_test_tot;
        vect_testing_error(i_runs) = testing_error;

        TP0 = 0;
        TP1 = 0;
        TP2 = 0;
        F01 = 0;
        F02 = 0;
        F10 = 0;
        F12 = 0;
        F20 = 0;
        F21 = 0;
        
        for user = 1:m_test_tot
            if label(user)==0 
                if predc(user)==0
                    TP0 = TP0 + 1;
                elseif predc(user)==1
                    F01 = F01 + 1;
                elseif predc(user)==2
                    F02 = F02 + 1;
                end
            elseif label(user)==1
                if predc(user)==1
                    TP1 = TP1 + 1;
                elseif predc(user)==0
                    F10 = F10 + 1;
                elseif predc(user)==2
                    F12 = F12 + 1;
                end
            elseif label(user)==2
                if predc(user)==2
                    TP2 = TP2 + 1;
                elseif predc(user)==0
                    F20 = F20 + 1;
                elseif predc(user)==1
                    F21 = F21 + 1;
                end
            end
        end
        vect_TP0(i_runs) = TP0;
        vect_TP1(i_runs) = TP1;
        vect_TP2(i_runs) = TP2;
        vect_F01(i_runs) = F01;
        vect_F02(i_runs) = F02;
        vect_F10(i_runs) = F10;
        vect_F12(i_runs) = F12;
        vect_F20(i_runs) = F20;
        vect_F21(i_runs) = F21;
    end
    toc

    matrice = zeros(3,3);
    matrice(1,1)=sum(vect_TP0);
    matrice(2,2)=sum(vect_TP1);
    matrice(3,3)=sum(vect_TP2);
    matrice(2,1)=sum(vect_F01);
    matrice(3,1)=sum(vect_F02);
    matrice(1,2)=sum(vect_F10);
    matrice(3,2)=sum(vect_F12);
    matrice(1,3)=sum(vect_F20);
    matrice(2,3)=sum(vect_F21);
    disp(matrice)

    % restituisco statistiche sul testing error
    fprintf('Tempo trascorso %.2f\n', toc);
    if exist('alpha', 'var')
        disp('gaussian kernel')
    else 
        fprintf('d=%d - c=%d\n', vettd{lap}, vettc{lap});
    end

    mean_all = mean(vect_testing_error);
    fprintf('mean testing accuracy %.2f\n', (1-mean_all)*100);
    std_all = std(vect_testing_error);
    fprintf('std testing error %.2f\n', std_all*100);

    sensitivity0 = matrice(1,1)/(sum(matrice(:,1)));
    fprintf('sensitivity0 %.2f\n', sensitivity0*100);
    sensitivity1 = matrice(2,2)/(sum(matrice(:,2)));
    fprintf('sensitivity1 %.2f\n', sensitivity1*100);
    sensitivity2 = matrice(3,3)/(sum(matrice(:,3)));
    fprintf('sensitivity2 %.2f\n', sensitivity2*100);

    specificity0 = sum(matrice(2:3,2:3))/sum(matrice(:,2:3));
    fprintf('specificity0 %.2f\n', specificity0*100);
    specificity1 = (sum(matrice(1,1))+sum(matrice(1,3))+sum(matrice(3,3))+sum(matrice(3,1)))/(sum(matrice(:,1))+sum(matrice(:,3)));
    fprintf('specificity1 %.2f\n', specificity1*100);
    specificity2 = sum(matrice(1:2,1:2))/sum(matrice(:,1:2));
    fprintf('specificity2 %.2f\n', specificity2*100);

    %MCC = ((sum(vect_TP)*sum(vect_TN))-(sum(vect_FP)*sum(vect_FN)))/sqrt((sum(vect_TP)+sum(vect_FP))*(sum(vect_TP)+sum(vect_FN))*(sum(vect_TN)+sum(vect_FP))*(sum(vect_TN)+sum(vect_FN)));
    %fprintf('MCC %.2f\n', MCC);
    %fprintf(fileID, 'MCC: %.2f\n', MCC);

    %ER = (sum(vect_FP)+sum(vect_FN))/(sum(vect_TP)+sum(vect_FN)+sum(vect_FP)+sum(vect_TN));
    %fprintf('ER %.2f\n', ER*100);
    %fprintf(fileID, 'ER: %.2f\n', ER*100);
    
    disp('TP0')
    disp(vect_TP0)
    disp('TP1')
    disp(vect_TP1)
    disp('TP2')
    disp(vect_TP2)
    disp('F01')
    disp(vect_F01)
    disp('F02')
    disp(vect_F02)
    disp('F10')
    disp(vect_F10)
    disp('F12')
    disp(vect_F12)
    disp('F20')
    disp(vect_F20)
    disp('F21')
    disp(vect_F21)
    disp('best nu')
    disp(vect_best_nu)
end

function error = calculate_error(nu, D_hat, K, m_train_tot)
    cvx_begin quiet
    cvx_solver mosek
    cvx_precision high
    variables u_l(m_train_tot) vargamma_l xi_l(m_train_tot) s_l(m_train_tot)
    minimize sum(s_l) + nu{1,1}*sum(xi_l)
    subject to
    D_hat*(K*D_hat*u_l-ones(m_train_tot,1)*vargamma_l)+xi_l >= ones(m_train_tot,1);
    xi_l >= 0;
    u_l >= -s_l;
    u_l <= s_l;
    s_l >= 0;
    cvx_end;
                
    % calcolo i valori di omega
    omega_minus_l = - min(D_hat*xi_l);
    omega_l = max(D_hat*xi_l);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%% risolvo la versione non lineare di Liu e Potra
    num_points = 1e2;
    discr_b_l = linspace(vargamma_l+1-omega_minus_l,vargamma_l-1+omega_l,num_points);

    max_b = m_train_tot;
    prova1 = -K*D_hat*u_l;
    for j = 1:length(discr_b_l)
        b_l = discr_b_l(j);
        prova2 = D_hat*(prova1+ones(m_train_tot,1)*b_l);
        if sum(prova2>0) < max_b
            max_b = sum(prova2>0);
            b_opt_l = b_l;
        end
    end                               
    % trovo quanti punti sono misclassified
    tot_num_misclass_training = length(find(D_hat*(-K*D_hat*u_l+ones(m_train_tot,1)*b_opt_l)>0));
    error = tot_num_misclass_training/m_train_tot; % average
end