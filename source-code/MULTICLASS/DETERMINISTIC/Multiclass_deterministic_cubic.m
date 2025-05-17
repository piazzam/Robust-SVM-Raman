% Main per il problema MULTICLASS deterministico

format long
%%%%%%%%%%%%%%%%%%%%%%%%%%%% pulizia
clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%% importo i dati
DATA = readtable("C:\Users\achic\OneDrive - unibg.it\Desktop\TESI Ricerca Operativa\pca-reduced\15_componenti.csv");
DATA = DATA(2:end,:);

% [num_righe, ~] = size(DATA);
% indici_casuali = randperm(num_righe, 50);
% DATA = DATA(indici_casuali, :);
% DATA = varfun(@mean, DATA, 'GroupingVariables', 'Var16');

users = unique(DATA.Var16);
%%%%%%%%%%%%%%%%%%%%%%%%%%%% testo il tutto per il numero di users distinci
vettd = {1,2,3,1,2,3};
vettc = {0,0,0,5,5,5};
n_runs = numel(users);
lista_pazienti1 = {1,2,3,4,5,6,7,8,9,10,31,32,33,34,35,36,37,38,39,40,41,42,68,69,70,71,72,73,74,75,76,77,78,79};
lista_pazienti2 = {11,12,13,14,15,16,17,18,19,20,43,44,45,46,47,48,49,50,51,52,53,54,80,81,82,83,84,85,86,87,88,89,90,91};
lista_pazienti3 = {21,22,23,24,25,26,27,28,29,30,55,56,57,58,59,60,61,62,63,64,65,66,67,92,93,94,95,96,97,98,99,100,101};

for lap = 6:6
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
    vect_best_nu = zeros(n_runs,3);

    % risolvo il modello    
    tic
    
    for p = 1:length(lista_pazienti3)
        i_runs = lista_pazienti3{p};
        fprintf('    iterazione %d/%d\n', p, length(lista_pazienti3));
        
        persona = users{i_runs};
        [vect_best_nu(i_runs,:),vect_TP0(i_runs),vect_TP1(i_runs),vect_TP2(i_runs),...
            vect_F01(i_runs),vect_F02(i_runs),vect_F10(i_runs),...
            vect_F12(i_runs),vect_F20(i_runs),vect_F21(i_runs),...
            vect_testing_error(i_runs)] = multiclass_deterministic_unit(DATA, persona,lap);        
    end   
    toc
    matrice = zeros(3,3);
    matrice(1,1)=sum(vect_TP0);
    matrice(2,2)=sum(vect_TP1);
    matrice(3,3)=sum(vect_TP2);
    matrice(2,1)=sum(vect_TN01);
    matrice(3,1)=sum(vect_TN02);
    matrice(1,2)=sum(vect_TN10);
    matrice(3,2)=sum(vect_TN12);
    matrice(1,3)=sum(vect_TN20);
    matrice(2,3)=sum(vect_TN21);
    disp(matrice)
    % restituisco statistiche sul testing error
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

    sensitivity0 = matrice(1,1)/(sum(matrice(:,1)));
    fprintf('sensitivity0 %.2f\n', sensitivity0*100);
    %fprintf(fileID, 'sensitivity: %.2f\n', sensitivity*100);
    sensitivity1 = matrice(2,2)/(sum(matrice(:,2)));
    fprintf('sensitivity1 %.2f\n', sensitivity1*100);
    %fprintf(fileID, 'sensitivity: %.2f\n', sensitivity*100);
    sensitivity2 = matrice(3,3)/(sum(matrice(:,3)));
    fprintf('sensitivity2 %.2f\n', sensitivity2*100);
    %fprintf(fileID, 'sensitivity: %.2f\n', sensitivity*100);

    specificity0 = sum(matrice(2:3,2:3))/sum(matrice(:,2:3));
    fprintf('specificity0 %.2f\n', specificity0*100);
    %fprintf(fileID, 'specificity: %.2f\n', specificity*100);
    specificity1 = (sum(matrice(1,1))+sum(matrice(1,3))+sum(matrice(3,3))+sum(matrice(3,1)))/(sum(matrice(:,1))+sum(matrice(:,3)));
    fprintf('specificity1 %.2f\n', specificity1*100);
    %fprintf(fileID, 'specificity: %.2f\n', specificity*100);
    specificity2 = sum(matrice(1:2,1:2))/sum(matrice(:,1:2));
    fprintf('specificity2 %.2f\n', specificity2*100);
    %fprintf(fileID, 'specificity: %.2f\n', specificity*100);
    
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
    disp('testing_error')
    disp(vect_testing_error)
end