% MY - pb robusto applicato al caso (DATASET), con training/testing set
format long
%%%%%%%%%%%%%%%%%%%%%%%%%%%% pulizia
clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%% importo i dati
DATA = readtable("15_componenti.csv");
DATA = DATA(2:end,:);
DATA = DATA(DATA.Var18 ~= 2, :);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% testo il tutto per n_runs volte (multiplo di 8 così da sfruttare il calcolo parallelo)
%n_runs = 96;

% vect_TP = zeros(n_runs,1);
% vect_FN = zeros(n_runs,1);
% vect_FP = zeros(n_runs,1);
% vect_TN = zeros(n_runs,1);
% vect_testing_error =   zeros(n_runs,1);
% vect_testing_error_A = zeros(n_runs,1);
% vect_testing_error_B = zeros(n_runs,1);

tic

%%% FOR PER AVERE TRUE/FALSE POSITIVE/NEGATIVE
% parfor i_runs = 1:n_runs
%     [vect_TP(i_runs), vect_FN(i_runs), vect_FP(i_runs), vect_TN(i_runs),...
%         vect_testing_error(i_runs)] = unit_of_work_ROBUST(DATA);
% end

%%% FOR PER AVERE SOLO TESTING ERROR SULL'INTERO TESTING SET
%parfor i_runs = 1:n_runs
%   [~, ~, ~, ~,...
%       vect_testing_error(i_runs)] = unit_of_work_ROBUST(DATA);
%end

%%% FOR PER AVERE TESTING ERROR SULL'INTERO TESTING SET, SULLA CLASSE A E
%%% SULLA CLASSE B
% parfor i_runs = 1:n_runs
%    [~, ~, ~, ~,...
%        vect_testing_error(i_runs),...
%        vect_testing_error_A(i_runs),...
%        vect_testing_error_B(i_runs)] = unit_of_work_ROBUST(DATA);
% end

%%% FOR PER AVERE TESTING ERROR SULL'INTERO TESTING SET, SULLA CLASSE A,
%%% SULLA CLASSE B E AL VARIARE DI RHO CHE VIENE PASSATO IN INGRESSO
users = unique(DATA.Var17);
n_runs = numel(users);
vect_rho = logspace(-7,-1,7);
%vect_rho = logspace(-7,-1,60);
num_rho = length(vect_rho);
vect_testing_error = zeros(n_runs,num_rho);
vect_testing_error_A = zeros(n_runs,num_rho);
vect_testing_error_B = zeros(n_runs,num_rho);


%for index_rho = 1:num_rho
for index_rho = 1:1
    scalar_rho = vect_rho(index_rho);
    disp(index_rho)
    for i_runs = 1:5
    %for i_runs = 1:n_runs
        persona = users{i_runs};
        disp(persona)
        [~, ~, ~, ~,...
            vect_testing_error(i_runs,index_rho),...
            vect_testing_error_A(i_runs,index_rho),...
            vect_testing_error_B(i_runs,index_rho)] = unit_of_work_ROBUST(DATA,scalar_rho,i_runs,n_runs,users,persona);
    end
end



toc

vect_testing_error;

disp('mean testing error')
mean_all = mean(vect_testing_error)';
% 
% disp('std testing error')
% std_all = std(vect_testing_error)
% 
% % MISS SU CLASSE A
% vect_testing_error_A;
% 
% disp('mean testing error class A')
% mean_classA = mean(vect_testing_error_A)
% 
% disp('std testing error class A')
% std_classA = std(vect_testing_error_A)
% 
% % MISS SU CLASSE B
% vect_testing_error_B;
% 
% disp('mean testing error class B')
% mean_classB = mean(vect_testing_error_B)
% 
% disp('std testing error class B')
% std_classB = std(vect_testing_error_B)


% figure
% confusionchart([sum(vect_TN) sum(vect_FP); sum(vect_FN) sum(vect_TP)],...
%    [-1 1])
% 
% disp('     TP   FN    FP   TN')
% disp([vect_TP vect_FN vect_FP vect_TN])
