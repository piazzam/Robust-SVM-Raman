% MY - pb nominale applicato al caso (DATASET), con training/testing set
format long
%%%%%%%%%%%%%%%%%%%%%%%%%%%% pulizia
clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%% importo i dati
DATA = readtable("15_componenti.csv");
DATA = DATA(2:end,:);
DATA = DATA(DATA.Var17 ~= 1, :); %inserisci la classe che non ti serve

users = unique(DATA.Var16); %inserisci il nome della variabile dove ci sono i nomi pazienti

%%%%%%%%%%%%%%%%%%%%%%%%%%%% testo il tutto per n_runs volte
n_runs = length(users);
vettc = {0,0,0,5,5,5};
vettd = {1,2,3,1,2,3};
%fileID = fopen('output.txt', 'w');

%i lap sono fatti per lanciare i 6 kernel in fila
for lap = 1:6
    % fprintf('lap %d/%d\n', lap, 6);
    vect_TP = zeros(n_runs,1);
    vect_FN = zeros(n_runs,1);
    vect_FP = zeros(n_runs,1);
    vect_TN = zeros(n_runs,1);
    vect_testing_error = zeros(n_runs,1);
    vect_best_nu = zeros(n_runs,1);
    
    tic

    %parfor i_runs = 1:numel(users) %usa questo se lanci in parallelo
    for i_runs = 1:numel(users)
        fprintf('    iterazione %d/%d\n', i_runs, n_runs);
        persona = users{i_runs};

        [vect_TP(i_runs), vect_FN(i_runs), vect_FP(i_runs), vect_TN(i_runs),...
            vect_testing_error(i_runs), vect_best_nu(i_runs)] = Binary_deterministic_unit(DATA,persona,lap);        
    end

    toc    
    %fprintf(fileID, 'Tempo trascorso %.2f\n', toc);
    fprintf('Tempo trascorso %.2f\n', toc);

    %fprintf('gaussian kernel');
    %fprintf(fileID, 'gaussian kernel');
    fprintf('d=%d - c=%d\n', vettd{lap}, vettc{lap});
    %fprintf(fileID, 'd=%d - c=%d\n', vettd{lap}, vettc{lap});

    mean_all = mean(vect_testing_error);
    fprintf('mean testing accuracy %.2f\n', (1-mean_all)*100);
    %fprintf(fileID, 'mean testing accuracy %.2f\n', (1-mean_all)*100);
    
    std_all = std(vect_testing_error);
    fprintf('std testing error %.2f\n', std_all*100);
    %fprintf(fileID, 'std testing error: %.2f\n', std_all*100);

    precision = sum(vect_TP)/(sum(vect_TP)+sum(vect_FP));
    fprintf('precision %.2f\n', precision*100);

    sensitivity = sum(vect_TP)/(sum(vect_TP)+sum(vect_TN));
    fprintf('sensitivity %.2f\n', sensitivity*100);
    %fprintf(fileID, 'sensitivity: %.2f\n', sensitivity*100);

    specificity = sum(vect_TN)/(sum(vect_TP)+sum(vect_TN));
    fprintf('specificity %.2f\n', specificity*100);
    %fprintf(fileID, 'specificity: %.2f\n', specificity*100);

    MCC = ((sum(vect_TP)*sum(vect_TN))-(sum(vect_FP)*sum(vect_FN)))/sqrt((sum(vect_TP)+sum(vect_FP))*(sum(vect_TP)+sum(vect_FN))*(sum(vect_TN)+sum(vect_FP))*(sum(vect_TN)+sum(vect_FN)));
    fprintf('MCC %.2f\n', MCC);
    %fprintf(fileID, 'MCC: %.2f\n', MCC);

    ER = (sum(vect_FP)+sum(vect_FN))/(sum(vect_TP)+sum(vect_FN)+sum(vect_FP)+sum(vect_TN));
    fprintf('ER %.2f\n', ER*100);
    %fprintf(fileID, 'ER: %.2f\n', ER*100);
    
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
end

%fclose(fileID);