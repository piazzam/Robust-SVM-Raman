function [best_nu_l,TP0,TP1,TP2,F01,F02,F10,F12,F20,F21,testing_error] = multiclass_deterministic_unit(DATA, persona,lap)        
        
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
        vectornu = [0.006003704206442 0.018746888575903 0.069310493609802...
                    0.534775938716443 1.433725286800503];

        u_vect = zeros(m_train_tot,L);
        b_vect = zeros(1,L);
        D_hat_tensore = zeros(m_train_tot,m_train_tot,L);    
        best_nu_l = zeros(1,3);

        for l=1:L
            %fprintf('        %d/%d\n', l, L);
            y_hat = -ones(m_train_tot,1);
            D_hat = zeros(m_train_tot,m_train_tot);
            y_hat = 1*(y_label==l-1)-1*(y_label~=l-1);
            D_hat = diag(y_hat);

            training_error_opt = Inf;

            for i_nu = 1:length(vectornu)
                %fprintf('    %d/%d\n',i_nu, length(vectornu))
                nu = vectornu(i_nu);

                cvx_begin quiet
                cvx_solver mosek
                cvx_precision high
                variables u_l(m_train_tot) vargamma_l xi_l(m_train_tot) s_l(m_train_tot)
                minimize sum(s_l) + nu*sum(xi_l)
                subject to
                D_hat*(K*D_hat*u_l-ones(m_train_tot,1)*vargamma_l)+xi_l >= ones(m_train_tot,1);
                xi_l >= 0;
                u_l >= -s_l;
                u_l <= s_l;
                s_l >= 0;
                cvx_end;
                %disp('      fatto cvx')
                
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

                training_error = tot_num_misclass_training/m_train_tot; % average

                if training_error < training_error_opt
                    training_error_opt = training_error;
                    u_opt_l = u_l;
                    b_opt_opt_l = b_opt_l;
                    selected_nu = nu;
                end
            best_nu_l(l) = selected_nu;
            end
            
            u_vect(:,l) = u_opt_l;
            b_vect(:,l) = b_opt_opt_l;
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
                if exist('alpha', 'var')
                    K_test(j) = exp(-norm(dati(:,j)-x_test)^2/(2*alpha^2));
                else
                    K_test(j) = (dati(:,j)'*x_test+c)^d;
                end
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