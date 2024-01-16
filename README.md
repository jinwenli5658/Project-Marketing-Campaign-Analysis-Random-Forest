# Project: Marketing Campaign Analysis (Random Forest)
Using Random Forest to predict if the client will respond on the marketing campaign </br>

## Data Imputation </br>
Imputing the missing values either using max/min.

## One Hot Coding </br>
Hard code, changing categorical variables to 1 or 0. </br>

## Random Forest </br>
from sklearn.ensemble import RandomForestClassifier </br>
rf_clf = RandomForestClassifier(max_depth=7, random_state=824) </br>
rf_clf.fit(X_train, y_train) </br>

## Feature Importance </br>
for name, importance in zip(X_train.columns.tolist(), rf_clf.feature_importances_): </br>
    print('%s = %.3f' %(name, importance)) </br>

## Model Evaluation: ROC/AUC Score </br>
from sklearn.metrics import roc_auc_score </br>
roc_auc_score(y_train, rf_clf.predict_proba(X_train)[:,1]) </br>
roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:,1]) </br>


## Data Columns: </br>
1 - cust_id : customer id </br>
2 - age (numeric) </br>
3 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed') </br>
4 - marital : marital status (categorical: 'divorced','married','single') </br>
5 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree') </br>
6 - default: has credit in default? (categorical: 'no','yes') </br>
7 - mortgage: has housing loan? (categorical: 'no','yes') </br>
8 - loan: has personal loan? (categorical: 'no','yes') </br>
9 - contact_type: contact communication type (categorical: 'cellular','telephone') </br>
10 - date: last contact date </br>
11 - duration: last contact duration, in seconds (numeric) </br>
12 - contact_num: number of contacts performed during this campaign and for this client (numeric, includes last contact) </br>
13 - p_days: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted) </br>
14 - p_outcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success') </br>
15 - y - has the client subscribed a term deposit? (binary: 'yes','no') </br>


