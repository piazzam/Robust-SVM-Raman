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

for lap = 5:5
    % fprintf('lap %d/%d\n', lap, 6);
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
    vect_best_nu = zeros(n_runs,3);
    vect_best_rho = zeros(n_runs,3);   
    
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
           
        % costruisco i dati sulle singole classi (tot 3 classi)
        dati_class_1 = DATAtrain(1:end-1,1:m_train(1));
        dati_class_2 = DATAtrain(1:end-1,m_train(1)+1:m_train(1)+m_train(2));
        dati_class_3 = DATAtrain(1:end-1,m_train(1)+m_train(2)+1:end);

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
        
        if p == inf
            C = sqrt(n);
        elseif p <= 2
            C = 1;
        elseif p > 2
            C = n^((p-2)/(2*p));
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
            % Definisci lo spazio di ricerca per nu
            nu_range = [0, 2];  % Intervallo di valori di nu
            var1 = optimizableVariable('nu', nu_range, 'Type', 'real');
            rho_range = [10^(-7), 10^(-1)];
            var2 = optimizableVariable('scalar_rho', rho_range, 'Type', 'real');
            vars = [var1,var2];
        
            % Definisci la funzione obiettivo per l'ottimizzazione bayesiana
            objective = @(vars) calculate_error(vars.nu, D_hat, K, std_1, std_2, std_3, m_train, m_train_tot, C,...
                dati_class_1, dati_cl12ass_2, dati_class_3, vars.scalar_rho, d, K_d_sqrt, c);
        
            % Configura l'ottimizzazione bayesiana
            results = bayesopt(objective, vars, UseParallel=false, AcquisitionFunctionName="probability-of-improvement", MaxObjectiveEvaluations=30, Verbose=1, PlotFcn=[]);
            % results = bayesopt(objective, vars, AcquisitionFunctionName="probability-of-improvement", UseParallel=false, Verbose=1, PlotFcn=[]);
            
            % Ottieni il miglior valore di nu
            best_nu = results.XAtMinObjective.nu;
            best_scalar_rho = results.XAtMinObjective.scalar_rho;
            vect_best_nu(i_runs,l) = best_nu;
            vect_best_rho(i_runs,l) = best_scalar_rho;

            rho_1 = best_scalar_rho;
            rho_2 = rho_1;
            rho_3 = rho_1;
            eta_1 = rho_1*max(std_1);
            eta_2 = rho_2*max(std_2);
            eta_3 = rho_3*max(std_3);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%% perturbazioni (FEATURE space)
            % costruzione della costante C
            
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
            if d > 1
                % per classe 1
                aux_j = 0;
                aux_k = 0;
                for i = 1:m_train(1)
                    x_i = dati_class_1(:,i);
                    for k = 1:d
                        delta_1(i) = delta_1(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_1)^k;
                    end
                    for k = 1:d-1
                        for j = 1:d-k
                            aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_1)^j;
                        end
                        aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
                    end
                    delta_1(i) = (delta_1(i))^2 + (aux_k)^2;
                end
                delta_1 = sqrt(delta_1);

                % per classe 2
                aux_j = 0;
                aux_k = 0;
                for i = 1:m_train(2)
                    x_i = dati_class_2(:,i);
                    for k = 1:d
                        delta_2(i) = delta_2(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_2)^k;
                    end
                    for k = 1:d-1
                        for j = 1:d-k
                            aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_2)^j;
                        end
                        aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
                    end
                    delta_2(i) = (delta_2(i))^2 + (aux_k)^2;
                end
                delta_2 = sqrt(delta_2);

                % per classe 3
                aux_j = 0;
                aux_k = 0;
                for i = 1:m_train(3)
                    x_i = dati_class_3(:,i);
                    for k = 1:d
                        delta_3(i) = delta_3(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_3)^k;
                    end
                    for k = 1:d-1
                        for j = 1:d-k
                            aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_3)^j;
                        end
                        aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
                    end
                    delta_3(i) = (delta_3(i))^2 + (aux_k)^2;
                end
                delta_3 = sqrt(delta_3);
                % costruisco il vettore dei delta
                delta = [delta_1' delta_2' delta_3'];
                delta = delta';
            end
            
            % % delta su RBF Kernel
            % delta_1 = sqrt(2-2*exp(-(C*eta_1)^2/(2*alpha^2)));
            % delta_2 = sqrt(2-2*exp(-(C*eta_2)^2/(2*alpha^2)));
            % delta_3 = sqrt(2-2*exp(-(C*eta_3)^2/(2*alpha^2)));
            % delta = [delta_1*ones(1,m_train(1)) delta_2*ones(1,m_train(2)) delta_3*ones(1,m_train(3))];
            % delta = delta';
            
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
    % if exist('alpha', 'var')
    %     disp('gaussian kernel')
    % else 
    %     fprintf('d=%d - c=%d\n', vettd{lap}, vettc{lap});
    % end

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
    disp('best rho')
    disp(best_scalar_rho)
end

function error = calculate_error(nu, D_hat, K, std_1, std_2, std_3, m_train, m_train_tot, C,...
                dati_class_1, dati_class_2, dati_class_3, scalar_rho, d, K_d_sqrt, c)
    
    rho_1 = scalar_rho;
    rho_2 = rho_1;
    rho_3 = rho_1;
    eta_1 = rho_1*max(std_1);
    eta_2 = rho_2*max(std_2);
    eta_3 = rho_3*max(std_3);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% perturbazioni (FEATURE space)
    % costruzione della costante C
    
    % % delta su polinomiale omogenea di grado d
    delta_1 = zeros(m_train(1),1);
    delta_2 = zeros(m_train(2),1);
    delta_3 = zeros(m_train(3),1);
    % if d == 1
    %     delta_1 = C*eta_1;
    %     delta_2 = C*eta_2;
    %     delta_3 = C*eta_3;
    %     % costruisco il vettore dei delta
    %     delta = [delta_1*ones(1,m_train(1)) delta_2*ones(1,m_train(2)) delta_3*ones(1,m_train(3))];
    %     delta = delta';
    % end
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
    if d > 1
        % per classe 1
        aux_j = 0;
        aux_k = 0;
        for i = 1:m_train(1)
            x_i = dati_class_1(:,i);
            for k = 1:d
                delta_1(i) = delta_1(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_1)^k;
            end
            for k = 1:d-1
                for j = 1:d-k
                    aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_1)^j;
                end
                aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
            end
            delta_1(i) = (delta_1(i))^2 + (aux_k)^2;
        end
        delta_1 = sqrt(delta_1);

        % per classe 2
        aux_j = 0;
        aux_k = 0;
        for i = 1:m_train(2)
            x_i = dati_class_2(:,i);
            for k = 1:d
                delta_2(i) = delta_2(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_2)^k;
            end
            for k = 1:d-1
                for j = 1:d-k
                    aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_2)^j;
                end
                aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
            end
            delta_2(i) = (delta_2(i))^2 + (aux_k)^2;
        end
        delta_2 = sqrt(delta_2);

        % per classe 3
        aux_j = 0;
        aux_k = 0;
        for i = 1:m_train(3)
            x_i = dati_class_3(:,i);
            for k = 1:d
                delta_3(i) = delta_3(i) + nchoosek(d,k)*norm(x_i,2)^(d-k)*(C*eta_3)^k;
            end
            for k = 1:d-1
                for j = 1:d-k
                    aux_j = aux_j + nchoosek(d-k,j)*norm(x_i,2)^(d-k-j)*(C*eta_3)^j;
                end
                aux_k = aux_k + nchoosek(d,k)*c^k*(aux_j)^2;
            end
            delta_3(i) = (delta_3(i))^2 + (aux_k)^2;
        end
        delta_3 = sqrt(delta_3);
        % costruisco il vettore dei delta
        delta = [delta_1' delta_2' delta_3'];
        delta = delta';
    end
    
    % % delta su RBF Kernel
    % delta_1 = sqrt(2-2*exp(-(C*eta_1)^2/(2*alpha^2)));
    % delta_2 = sqrt(2-2*exp(-(C*eta_2)^2/(2*alpha^2)));
    % delta_3 = sqrt(2-2*exp(-(C*eta_3)^2/(2*alpha^2)));
    % delta = [delta_1*ones(1,m_train(1)) delta_2*ones(1,m_train(2)) delta_3*ones(1,m_train(3))];
    % delta = delta';

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
                
    % calcolo i valori di omega
    omega_minus_l = - min(D_hat*xi_l);
    omega_l = max(D_hat*xi_l);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%% risolvo la versione non lineare di Liu e Potra
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
    tot_num_misclass_training = length(find(D_hat*(-K*D_hat*u_l+ones(m_train_tot,1)*b_opt_l)>0));
    error = tot_num_misclass_training/m_train_tot; % average
end