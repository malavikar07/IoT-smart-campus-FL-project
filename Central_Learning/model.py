import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

classroom=pd.read_csv('Training/dataset/Classroom.csv')
Computer_lab=pd.read_csv('Training/dataset/Computer_lab.csv')
Hallway=pd.read_csv('Training/dataset/Hallway.csv')
Testing=pd.read_csv('Training/dataset/Testing_data.csv').drop(columns=["timestamp","room","day_of_week"])

training_data=pd.concat([pd.concat([classroom,Computer_lab]),Hallway]).drop(columns=["timestamp","room","day_of_week"])

X_train=training_data.drop(columns=['anomaly_lable'])
y_train=training_data['anomaly_label']

rf=RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_split=20)

rf.fit(X_train,y_train)

X_test=Testing.drop(columns=['anomaly_label'])
y_test=Testing['anomaly_label']

pred=rf.predict(X_test)

print("Ensembled Prediction accuracy:", accuracy_score(y_test,pred))
print("Emsembled Prediction Precision: ",precision_score(y_test,pred)) #Of all predicted anomalies, how many were actually correct?
print("Emsembled Prediction Recall: ",recall_score(y_test,pred)) #Of all actual anomalies, how many did we detect?
print("Emsembled Prediction F1 Score: ",f1_score(y_test,pred)) #Harmonic mean of precision and recall.
print("Emsembled Prediction AUC: ",roc_auc_score(y_test,pred)) #Tells how well your model distinguishes between classes across all thresholds.


