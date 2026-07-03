import pandas as pd
from sklearn.impute import SimpleImputer
from joblib import load
import numpy as np

room_id={0:'Classroom',1:'Computer lab',2:'Hallway'}

models = load('model_v1.0.joblib')
df=pd.read_csv('sensordata - sensor_log.csv')
df['delta_temp']=df['temperature_sensor'].diff()
imp=SimpleImputer(strategy='mean')
df['delta_temp']=imp.fit_transform(df[['delta_temp']])
for i in range(len(df)):
    row_df = df.iloc[[i]]  # double brackets to keep it as DataFrame
    X=row_df.drop(columns=['Date','room'])
    info=row_df[['Date','room']]
    preds = np.array([model.predict(X) for model in models])
    final_preds = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=preds)
    # print(final_preds)
    probas = np.array([model.predict_proba(X)[:, 1]
                       for model in models])
    ensemble_proba = probas.mean(axis=0)
    
    with open("threshold.txt","r") as f:
        threshold = float(f.read().strip())
    final_preds = (ensemble_proba >= threshold).astype(int)
    # print(final_preds)
    if final_preds[0]==1:
        room_key = info['room'].iloc[0]      # directly the int value
        timestamp = info['Date'].iloc[0] 
        print(f"Anomaly detected in {room_id[room_key]} at {timestamp}")
        _=input()
    else:
        print("No anomaly detected")