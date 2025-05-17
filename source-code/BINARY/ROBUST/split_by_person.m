function [Xtrain, Ytrain, Xtest, Ytest] = split_by_person(DATA, persona)
% SPLIT_BY_PERSON divide i dati in training e test set basandosi su una persona specifica.
%
% INPUT:
%   - DATA: tabella contenente le osservazioni con almeno due colonne:
%       * Var331: identificatore persona
%       * Var332: etichetta/classe (1 = target, 2 = input)
%   - persona: identificatore della persona da usare solo nel test set
%
% OUTPUT:
%   - Xtrain: input del training set (Var332 == 2, persone diverse da 'persona')
%   - Ytrain: target del training set (Var332 == 1, persone diverse da 'persona')
%   - Xtest: input del test set (Var332 == 2, solo della persona specificata)
%   - Ytest: target del test set (Var332 == 1, solo della persona specificata)

    % --- Training set: escludi la persona specificata
    trainingData = DATA(DATA.Var331 ~= string(persona), :);
    trainingData.Var331 = []; % rimuove colonna persona

    % Separa input e target per il training
    XtrainTable = trainingData(trainingData.Var332 == 2, :);
    YtrainTable = trainingData(trainingData.Var332 == 1, :);
    
    XtrainTable.Var332 = [];  % rimuove etichetta dai dati input
    YtrainTable.Var332 = [];  % rimuove etichetta dai dati target

    % --- Test set: dati solo della persona specificata
    testData = DATA(DATA.Var331 == string(persona), :);
    testData.Var331 = []; % rimuove colonna persona

    XtestTable = testData(testData.Var332 == 2, :);
    YtestTable = testData(testData.Var332 == 1, :);

    XtestTable.Var332 = [];  % rimuove etichetta dai dati input
    YtestTable.Var332 = [];  % rimuove etichetta dai dati target

    % --- Conversione da table a array
    Xtrain = table2array(XtrainTable);
    Ytrain = table2array(YtrainTable);
    Xtest = table2array(XtestTable);
    Ytest = table2array(YtestTable);
end