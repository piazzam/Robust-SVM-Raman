function [DATAtrain, DATAtest] = my_tries(DATA, testingsamplesize)
% funzione che effettua il partizionamento dei dati in due gruppi (training
% set e testing set), il secondo formato dal testingsamplesize% delle
% osservazioni totali

dp = cvpartition(DATA.CLASS, 'HoldOut', testingsamplesize);

idxtrain = training(dp);
DATAtrain = DATA(idxtrain,:);
DATAtrain = table2array(DATAtrain);

idxtest = test(dp);
DATAtest = DATA(idxtest,:);
DATAtest = table2array(DATAtest);


end