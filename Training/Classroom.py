import flwr as fl
from diffprivlib.models import RandomForestClassifier as DPRandomForest
import pickle
# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(filename='pipeline.log', level=logging.INFO)
# Load your per‑client CSV (sharded) as before
try:
    df = pd.read_csv("Training/dataset/Classroom.csv").drop(columns=["timestamp","room","day_of_week"])
    # df['value_diff'] = df['temperature_sensor'].diff()
    # from sklearn.impute import SimpleImputer
    # imputer=SimpleImputer(strategy='mean')
    # df['value_diff']=imputer.fit_transform(df[['value_diff']])
    # from sklearn.preprocessing import StandardScaler
    # df['temperature_sensor']=StandardScaler().fit_transform(df['temperature_sensor'])
    X, y = df.drop(columns="anomaly_label"), df["anomaly_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except Exception as e:
    logging.error(f"File issue: {str(e)}")

class DP_RFClient(fl.client.NumPyClient):
    def __init__(self):
        # ε is your privacy budget per client; adjust to taste
        self.model = DPRandomForest(
            n_estimators=225,
            max_depth=12,
            epsilon=0.5,
            data_norm=10.0,    # L2 sensitivity bound on features
            random_state=42
        )
        # Local DP training
        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")

    def get_parameters(self, config):
        # Serialize the entire DP‑RF model
        return [pickle.dumps(self.model)]

    def fit(self, parameters, config):
        # Receive global model, unpickle, then refit for one round
        global_model = pickle.loads(parameters[0])
        global_model.fit(X_train, y_train)
        return [pickle.dumps(global_model)], len(X_train), {}

    def evaluate(self, parameters, config):
        model = pickle.loads(parameters[0])
        loss = 1 - model.score(X_test, y_test)
        return float(loss), len(X_test), {"accuracy": model.score(X_test, y_test)}

if __name__ == "__main__":
    try:
        fl.client.start_numpy_client(server_address="localhost:8080", client=DP_RFClient())
    except Exception as e:
        logging.error(f"Server failed: {str(e)}")