# Robust-SVM-Raman

### Source code

The codes provided in this folder are related to the work "A Robust Support Vector Machine Approach for Raman Data Classification" (see the reference below).

All the codes are written in MATLAB. The models are solved using CVX and MOSEK. Please visit https://cvxr.com/cvx/ and https://www.mosek.com for details and licensing issues.

We provide four different folders, depending on the choice of the classifier (deterministic vs robust; binary vs multiclass). Specifically, each folder contains the following files:

- a main code in .m: the code that has to be run in MATLAB;
- a unit of work code in .m: the user has to properly choose the kernel function and the relevant parameters of the implementation;
- a leave-one-patient-out training-testing file in .m: a code to split the dataset into training set and testing set, by iteratively select a single patient as test set.

For all the details of the implementation, the user is referred to the reference reported as follows.

### Data

Saliva samples, health records, and clinical data were acquired at IRCCS Fondazione Don Carlo Gnocchi ONLUS, Santa Maria Nascente Hospital, Milano, Italy, and Centro Spalenza Hospital, Rovato, Italy between 16th April 2020 to July 2020 (Further information regarding the acquisition protocol and the dataset are available in the original paper: ["COVID-19 salivary Raman fingerprint: innovative approach for the detection of current and past SARS-CoV-2 infections"](https://www.nature.com/articles/s41598-021-84565-3)).

### Reference
