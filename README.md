The code is structured as follows. Code_5_Final is the main code, which calls other subfunctions. First the data es extracted from Code_1_*. This data is separated in train and test sets by Code_2_fun, then the optimization problem
on the whole grid of epsilon selected is solved in Code_4_**_Final (depending on the prior used) and saved for each fold. To solve the problem on each value of epsilon, an additional Code_3 ** is used inside each Code_4 to actually
solve the optimization problem via GlobalSearch+fmincon.

If one wants to work with the Student or Crime dataset, one must use Code_1_Student or Code_1_CRIME. 
If one wants to work with simulations, then it must use Code_1_Final. In this Code_1_Final, one can set the number of data points, the number of parameters and the way data is generated in general.

Additionally, in Code_5_Final one can set certain optimization parameters for the optimization solver, the number of folds (Num), the grid of epsilons on which the problem will be solved and on which set must the fairness constraint be imposed.

The standard version of the code is set to use simulated data with N=1000 and p=3, 2 folds and with the fairness constraint enforced in the test set.

**To obtain results for the Code_1 selected, one would only have to run Code_5_Final changing the corresponding names for the saved data to whichever names one wants**.
If it is desired to obtain results for different datasets, one must create a new Code_1. The requisites for this Code_1 is that it must deliver 3 things:

First it must deliver the whole design matrix (denoted by X0 in the code). With dimensions Nx(p+1). 
Second, it must deliver the vector of observations (denoted by y0 in the code). With dimensions Nx1.
Third, it must deliver a vector (denoted by isnotSens in the code) which takes values of 0 or 1. This vector has dimensions Nx1 and for each value i between 1 and N, the i-th value of vector isnotSens must be 1 if the i-th data point
is from a member of the non-sensitive class and 0 if it is a member of the sensitive class.

Afterwards, the train and test set are created in Code_2_fun_CV, as well as the values of z and zT. It is important to note that even though isnotSens is given by a certain Code_1, we only need it so that in Code_2_fun the values
of z and zT (z for the train and test set respectively) can be generated. If in reality the vector isnotSens is not known but the value of z and/or zT are known, the code can be modified in turn to bypass this order of creation
and use these values. The code is created in this way simply because the data used for the paper facilitated this way of usage.
