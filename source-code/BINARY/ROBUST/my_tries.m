function [Xtrain, Xtest, Ytrain, Ytest] = my_tries(DATA,persona,labels_a, labels_b)
    % funzione che effettua il partizionamento dei dati in due gruppi (training
    % set e testing set), il secondo formato dal testingsamplesize% delle
    % osservazioni totali
       
    DATAtrain = DATA(DATA.Var111 ~= string(persona), :);
    DATAtrain = removevars(DATAtrain, {'Var111'});

    
    X = DATAtrain(DATAtrain.Var112==labels_a, :);
    Y = DATAtrain(DATAtrain.Var112==labels_b, :);
    
    X(:,end)=[];
    Y(:,end)=[];
    
    DATAtest = DATA(DATA.Var111 == string(persona), :);
    DATAtest = removevars(DATAtest, {'Var111'});
    
    Xtest = DATAtest(DATAtest.Var112 == labels_a, :);
    Ytest = DATAtest(DATAtest.Var112 == labels_b, :);
    
    Xtest(:,end)=[];
    Ytest(:,end)=[];
    Xtrain=table2array(X);
    Ytrain=table2array(Y);
    Xtest=table2array(Xtest);
    Ytest=table2array(Ytest);
end

