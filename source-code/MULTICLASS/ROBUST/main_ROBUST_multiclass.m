% Main per il problema MULTICLASS robusto
% 3 febbraio 2024

format long
%%%%%%%%%%%%%%%%%%%%%%%%%%%% pulizia
clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%% importo i dati
DATA = readtable("15_componenti.csv");
DATA = DATA(2:end,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% testo il tutto per n_runs volte
users = unique(DATA.Var17);
n_runs = numel(users);
vect_rho = logspace(-7,-1,7);
% vect_rho = 1e-7;
num_rho = length(vect_rho);
vect_testing_error = zeros(n_runs,num_rho);

% risolvo il modello
tic
for index_rho = 1:num_rho
%for index_rho = 1:1
    fprintf('iterazione %d/%d\n', index_rho, num_rho);
    scalar_rho = vect_rho(index_rho);
    %disp(scalar_rho);
    for i_runs = 1:n_runs
        [vect_testing_error(i_runs,index_rho)] = uow_ROBUST_multiclass(DATA,scalar_rho,i_runs,n_runs,users);
    end
end
toc

% restituisco statistiche sul testing error
vect_testing_error;
%disp(vect_testing_error(1:5,:))
%disp('mean testing error')
mean_all = mean(vect_testing_error)';
disp(1-mean_all);
%for i = 1:5
%   disp(mean(vect_testing_error(i,:)))
%end