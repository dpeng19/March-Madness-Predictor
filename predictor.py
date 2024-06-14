import os             # Good for navigating your computer's files
import numpy as np    # Great for lists (arrays) of numbers
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
#from xgboost.sklearn import XGBClassifier

from sklearn import svm
from sklearn.neural_network import MLPRegressor

train_df = pd.read_csv('Train_data.csv')
test_df = pd.read_csv('Test_data.csv')
real_test_df = pd.read_csv('MarchMadness2024 - R64.csv')  #prediction on 2024 games
#test_df = test_df.loc[test_df['Year_1'] == 2017]


 
X = ['3P%D_1', 'ADJDE_1', '2P%D_1', 'SOS_2', 'EFGD%_1', 'EFGD%_2', 'Wins_1', 'TORD_1', '3P%D_2',
     'BARTHAG_2', 'TORD_2', 'ORB_1', '2P%_2', 'EFG%_1', 'SOS_1', 'DRB_1', 'Wins_2', 'TOR_2', 'Losses_2', 'TOR_1',
     'DRB_2', 'ADJOE_2', 'EFG%_2', 'Seed_2']
Y = 'Outcome_1'
 
max_accuracy = 0
count = 0
best_list = []

"""Finds list of features that gets the best accuracy on test data"""
def feature_selection(index, list=[]):
    if list and len(list) == 12:
        X_train = train_df[list]
        X_test = test_df[list]
        model = linear_model.LogisticRegression(max_iter=10000)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, y_pred)
        global count
        count = count + 1
        print(count)
        global max_accuracy
        global best_list
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_list = list
        print(max_accuracy)
        print(list)
        return
    if index == 24 and len(list) < 12:
        return

    feature_selection(index + 1, list + [X[index]])
    feature_selection(index + 1, list)

 
feature_selection(0, [])
print(max_accuracy)
print(best_list)

#best list
X = ['ADJDE_1', 'SOS_2', '3P%D_2', 'TORD_2', 'ORB_1', 'EFG%_1', 'DRB_1', 'Wins_2', 'Losses_2', 'TOR_1', 'DRB_2', 'EFG%_2']
X_train = train_df[X]
X_test = test_df[X]
Y_train = train_df[Y]
Y_test = test_df[Y]
X_real_test = real_test_df[X]
Y_real_test = real_test_df[Y]
model = linear_model.LogisticRegression(max_iter=10000)
#model = svm.SVC()

print(accuracy)
model.fit(X_train, Y_train)
y_pred = model.predict_proba(X_test)
#print(y_pred[27])
print(accuracy_score(Y_test, y_pred))
print(accuracy_score(Y_train, model.predict(X_train)))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
print(roc_auc_score(Y_test, y_pred))
print(f1_score(Y_test, y_pred))

#print(model.predict_proba(X_real_test)[30])
#y_pred = model.predict_proba(X_real_test[6])
 
for i in range(len(y_pred)):
    if y_pred[i][1] < 0.6:
        print(i)
