function [Xtrain, Xtest, Ytrain, Ytest] = my_tries(DATA,persona)
    % funzione che effettua il partizionamento dei dati in due gruppi (training
    % set e testing set), il secondo formato dal testingsamplesize% delle
    % osservazioni totali
       
    DATAtrain = DATA(DATA.Var16 ~= string(persona), :); %inserisci nome colonna con nome paziente
    DATAtrain = removevars(DATAtrain, {'Var16'}); %rimuovi quella colonna
    
    X = DATAtrain(DATAtrain.Var17==2, :); %inserisci colonna con le classi
    Y = DATAtrain(DATAtrain.Var17==1, :);
    
    X(:,end)=[];
    Y(:,end)=[];
    
    DATAtest = DATA(DATA.Var16 == string(persona), :);
    DATAtest = removevars(DATAtest, {'Var16'});
    
    Xtest = DATAtest(DATAtest.Var17 == 2, :);
    Ytest = DATAtest(DATAtest.Var17 == 1, :);
    
    Xtest(:,end)=[];
    Ytest(:,end)=[];
    Xtrain=table2array(X);
    Ytrain=table2array(Y);
    Xtest=table2array(Xtest);
    Ytest=table2array(Ytest);
end