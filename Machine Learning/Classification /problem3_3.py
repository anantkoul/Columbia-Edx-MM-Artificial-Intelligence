import numpy as np
import sys
import csv
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn import svm, linear_model, neighbors, tree, ensemble

input_data = np.genfromtxt(sys.argv[1], delimiter = ",")

Models = [['svm_linear'], ['svm_polynomial'], ['svm_rbf'], ['logistic'],['knn'], ['decision_tree'], ['random_forest']]

X_input_data = input_data[:,:2]
label = input_data[:,2]

CV_num = 5

Cross_validation = StratifiedShuffleSplit(n_splits = 5, test_size = 0.4, random_state = 0)




def classifer_fn(est_model, parameters, Cross_validation, X_input_data, label):
    clf = GridSearchCV(est_model, parameters, cv = Cross_validation, return_train_score = True)
    clf.fit(X_input_data, label)
    temp = clf.cv_results_
    test_m = 0
    train_m = 0
    for value in range(CV_num):
        k = 'split' + str(value) + '_train_score'
        p = 'split' + str(value) + '_test_score'
        #print('1')
        if max(temp[k]) > train_m:
            train_m = max(temp[k])
            #print('0')
        else:
            break
        if max(temp[p]) > test_m:
            test_m = max(temp[p])
            #print('2')
        else:
            break

    return [train_m,test_m]

#SVM with linear kernel
parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
est_model = svm.SVC(kernel = 'linear')
out = classifer_fn(est_model, parameters, Cross_validation, X_input_data, label)
Models[0] = Models[0] + out

# SVM with polynomial kernel
parameters = {'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma': [0.1, 0.5]}
est_model = svm.SVC(kernel = 'poly')
out = classifer_fn(est_model, parameters, Cross_validation, X_input_data, label)
Models[1] = Models[1] + out


# SVM with rbf
parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10]}
est_model = svm.SVC(kernel = 'rbf')
out = classifer_fn(est_model, parameters, Cross_validation, X_input_data, label)
Models[2] = Models[2] + out

# Logistic Regression
parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
est_model = linear_model.LogisticRegression()
out = classifer_fn(est_model, parameters, Cross_validation, X_input_data, label)
Models[3] = Models[3] + out

# KNN
parameters = {'n_neighbors': [i for i in range(1,50)], 'leaf_size': [5, 10, 15, 20, 25, 30, 60]}
est_model = neighbors.KNeighborsClassifier()
out = classifer_fn(est_model, parameters, Cross_validation, X_input_data, label)
Models[4] = Models[4] + out

# Decision tree
parameters = {'max_depth': [i for i in range(1,50)], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
est_model = tree.DecisionTreeClassifier()
out = classifer_fn(est_model, parameters, Cross_validation, X_input_data, label)
Models[5] = Models[5] + out

# Random Forest
parameters = {'max_depth': [i for i in range(1,50)], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
est_model = ensemble.RandomForestClassifier()
out = classifer_fn(est_model, parameters, Cross_validation, X_input_data, label)
Models[6] = Models[6] + out



with open ('output3.csv', 'w') as m:
    pub = csv.writer(m, delimiter = ',')
    for range in Models:
        pub.writerow(range)

#print(Models)
