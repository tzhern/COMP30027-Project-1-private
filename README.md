# COMP30027 Machine Learning Assignment 1
## Group Members
1. Tom Zhi Hern 1068268
2. Peter Qian Ziyu 1067810
 
## =====Naive Bayes Instructions=====
1. Read train and test data from _data/train.csv_ and _data/test.csv_ and convert them to dataframe (df_train & df_test)
2. Run _preprocess(df_train, df_test)_ to preprocess data and convert _df_test_ into numpy array (_np_test_)
3. Run _train(df_train)_ to get statistics from train data (prior probability, mean, stdv) and store it into _pose_dict_
4. Run _predict(np_test, pose_dict)_ to predict the results and store the results into list (_results_)
5. Run _evaluate(results)_ to get the accuracy of the model

## Question 1
1. Run _get_con_matrix(results, poses)_ to get confusion matrix (_con_matrix_) for the result obtained in the previous section
2. Run _print_model_eval(con_matrix)_ to evaluate the model, using micro and macro averaging and print the evaluation

## Question 2
1. Load all data from _data/all.csv_ which combined both data from _data/train.csv_ & _data/test.csv_ and store it into dataframe (_df_all_)
2. Add headers to _df_all_
3. Run _plot_qq(df_all, pose, remove=True)_ to plot QQ plot for each attribute (x1 to y11) for given pose
4. Choose pose from [mountain, downnwarddog, childs]

## Question 3
### Run cell_q3a then cell_q3b
#### cell_q3a
1. Read train and test data from _data/train.csv_ and _data/test.csv_ and convert them to dataframe (df_train & df_test)
2. Run _preprocess(df_train, df_test)_ to preprocess data and convert _df_test_ into numpy array (_np_test_)
3. Run _predict_kde(np_test, df_train, SIGMA=i)_ 2 times (sigma = 0.1 and sigma = 5) with a for loop
4. It will also run _get_con_matrix(results, poses)_ to get the confusion matrix for each result and print them

#### cell_q3b
1. Run this cell to plot pdf for gaussian & kde (with given sigma) for train dataset
2. Repeat with sigma=0.1 and sigma=0.5

## Question 4
1. Read train and test data from _data/train.csv_ and _data/test.csv_ and convert them to dataframe (df_train & df_test)
2. Run _predict_kde_rs(df, num)_ to run KDE Naive Bayes prediction random holdout using random holdout with given num. 5 is used here.
3. The result for each prediction will be printed out