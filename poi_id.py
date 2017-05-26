#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest,SelectPercentile,f_classif,chi2
from sklearn.decomposition import PCA
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
"""FEATURES SELECTED"""
features_list =  ['poi','salary', 'deferral_payments',  \
'bonus', 'restricted_stock_deferred', 'deferred_income',  'total_stock_value',\
'expenses','exercised_stock_options', 'other', 'long_term_incentive',\
'restricted_stock','director_fees', 'to_messages', 'from_poi_to_this_person',\
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
# You will need to use more features
#'total_payments',
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers (OUTLIER REMOVED)
data_dict.pop("TOTAL",0)
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#SCALE FEATURES
scaler=MinMaxScaler()
features = scaler.fit_transform(features)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
"""SELECTION OF THE APPROPRIATE FEATURE SELECTION TOOLS, SOME ARE COMMENTED
SINCE THEY COULD NOT PROVIDE THE BEST RESULTS"""

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()'salary'
#transform=SelectPercentile(f_classif)
transform=SelectKBest(f_classif)#Accuracy: 0.84780	Precision: 0.40806	Recall: 0.31400	F1: 0.35490	F2: 0.32917
 #Accuracy: 0.85213	Precision: 0.42838	Recall: 0.32600	F1: 0.37024	F2: 0.34237 when 'total_payments', is removed
pipe=Pipeline([("method",transform),("classifier",GaussianNB())])
#pipe=Pipeline([("method",transform),("classifier",MultinomialNB())])
#pipe=Pipeline([("method",transform),("classifier",LogisticRegression())])
#pipe=Pipeline([("method",PCA()),("classifier",GaussianNB())])
#pipe=Pipeline([("method",PCA()),("classifier",SVC())])
#pipe=Pipeline([("method",transform),("classifier",SVC())])
#LIST OF PARAMETERS
percentiles=(5,10,15,20,25,30,35,40,45,50,55,60,70,80,90,100)
penality= ('l1', 'l2')
CC=(0.01, 0.1, 1, 10)
n_components=(2,5,10,15)
Kbest=(5,8,10,12,15,16)
svm_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
param_gaussian=dict(method__k=Kbest)#dict(method__percentile=percentiles) #Precision ~0.34 obtained
params_logistic=dict(method__k=Kbest,classifier__penalty=penality,
classifier__C = CC) #very bad precsion and recall
param_pca=dict(method__n_components=n_components) #Best precision obtained
#Precision: 0.52788	Recall: 0.27450	F1: 0.36118	F2: 0.30365 (PCA + GgaussianNB())
#Accuracy: 0.86667	Precision: 0.50000	Recall: 0.28250	F1: 0.36102	F2: 0.30942 when 'total_payments', is removed
# Accuracy: 0.86600	Precision: 0.49548	Recall: 0.27400	F1: 0.35287	F2: 0.30090 when 'loan_advances', is removed
param_pca_svm=dict(method__n_components=n_components,classifier__C=CC) ## Very bad 0 precision and recall


param_kbest_svm=dict(method__k=Kbest,classifier__C=CC)
#clf=GridSearchCV(pipe, param_grid=param_pca,n_jobs=-1,verbose=0)
clf=GridSearchCV(pipe, param_grid=param_gaussian,n_jobs=-1,verbose=0)
#clf=GridSearchCV(pipe, param_grid=params_logistic,n_jobs=-1,verbose=0)
#clf=GridSearchCV(pipe, param_grid=params_logistic,n_jobs=-1,verbose=0)
#clf=GridSearchCV(pipe, param_grid=param_kbest_svm,n_jobs=-1,verbose=0)
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross valiudation. For more info:
### http://scikit-learn.org/stable/modles/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
"""SPLIT THE DATA INTO TRAINING AND TESTING SETS,THEN FIT AND PREDICT"""
from sklearn.cross_validation import train_test_split,cross_val_score
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

target_names=["poi","non_poi"]

clf.fit(features_train,labels_train)

pred=clf.predict(features_test)

print "The best estimater =",clf.best_estimator_

print classification_report(labels_test,pred, target_names=target_names)

print confusion_matrix(labels_test,pred)

print ("The scores from cross_val  on training data.......   ----->>>>>>")

precision=cross_val_score(clf, features_train,labels_train,cv=5,scoring="precision")
recall=cross_val_score(clf, features_train,labels_train,cv=5,scoring="recall")
f1=cross_val_score(clf, features_train,labels_train,cv=5,scoring="f1")
print("Precision="),np.mean(precision)
print ("Recall="), np.mean(recall)
print("F1_score="),np.mean(f1)
print("Best score: %0.3f" % clf.best_score_)
#print("Best score: %0.3f" % clf.best_scores_)
print ("-------->>>>>>>>>The scores from cross_val  on testing  data--->>>>>>")

print("precision on test data ="),precision_score(labels_test,pred)

print("Recall score on test data="),recall_score(labels_test,pred)
print("**************====================================*********************")
K_best = SelectKBest(f_classif, k=8)

features_kbest = K_best.fit_transform(features, labels)

print "Shape of features after applying SelectKBest -> ", features_kbest.shape
K_best = K_best

# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
#feature_scores = ['%.2f' % elem for elem in K_best.scores_ ]
# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in K_best.scores_ ]
print("==========================The feature scores list= ============")
print feature_scores
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "
#feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  K_best.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in'K_best.get_support',
# create a tuple of feature names, scores and pvalues, name it
#"features_selected_tuple"

features_selected_tuple=[(features_list[i+1], feature_scores[i], \
feature_scores_pvalues[i]) for i in K_best.get_support(indices=True)]
print("==========================Feature Tuple===============================")

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature:\
float(feature[1]) , reverse=True)

print features_selected_tuple
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
