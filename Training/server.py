import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.server import ServerConfig
import pickle
import numpy as np
import logging
from joblib import dump, load
import pandas as pd
from sklearn.metrics import precision_recall_curve
logging.basicConfig(filename='pipeline.log', level=logging.INFO)

def print_results(y_test,final_preds):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
    print("Ensembled Prediction accuracy:", accuracy_score(y_test,final_preds))
    print("Emsembled Prediction Precision: ",precision_score(y_test,final_preds)) #Of all predicted anomalies, how many were actually correct?
    print("Emsembled Prediction Recall: ",recall_score(y_test,final_preds)) #Of all actual anomalies, how many did we detect?
    print("Emsembled Prediction F1 Score: ",f1_score(y_test,final_preds)) #Harmonic mean of precision and recall.
    print("Emsembled Prediction AUC: ",roc_auc_score(y_test,final_preds)) #Tells how well your model distinguishes between classes across all thresholds.

try:
    df=pd.read_csv("Training/dataset/Testing_data.csv")
    df=df.drop(columns=["timestamp","room","day_of_week"])
    X_test=df.drop(columns="anomaly_label")
    y_test=df['anomaly_label'] # match feature count
except Exception as e:
        logging.error(f"File issue: {str(e)}")

class EnsembleStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ensemble_models = []

    def aggregate_fit(self,rnd, results,failures):
        # results: List[(client_proxy, FitRes), ...]
        models = []
        for _, fit_res in results:
            # fit_res.parameters is a Parameters object
            # Convert to raw tensors (list of pickled bytes)
            from flwr.common import parameters_to_ndarrays

            tensors = parameters_to_ndarrays(fit_res.parameters)
            # We serialized the entire RF model as a single pickle in client.get_parameters()
            model = pickle.loads(tensors[0])
            models.append(model)

        # Store the ensemble
        self.ensemble_models = models

        # Return first model's pickled bytes as the "global" model
        # (FedAvg expects you to return a Parameters-like sequence of tensors)
        return fit_res.parameters,{}  # send the last client's parameters back

def FL_training():
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=ServerConfig(num_rounds=3),
            strategy=strategy,
        )
    except Exception as e:
        logging.error(f"Server failed: {str(e)}")

def load_model_and_give_output(X_test):

    if strategy.ensemble_models:
        preds = np.array([model.predict(X_test) for model in strategy.ensemble_models])
        final_preds = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=preds)

        probas = np.array([model.predict_proba(X_test)[:, 1]
                       for model in strategy.ensemble_models])
    # Simple average (or replace with weighted average)
        ensemble_proba = probas.mean(axis=0)

    # 3) Compute precision–recall curve
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, ensemble_proba)

    # 4) (Optional) Plot it
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)

# Find the threshold with highest F1
        best_idx = f1_scores.argmax()
        best_threshold = thresholds[best_idx]
        best_precision = precision_vals[best_idx]
        best_recall = recall_vals[best_idx]
        best_f1 = f1_scores[best_idx]

        dump(strategy.ensemble_models, 'model_v1.0.joblib')  # save

        with open('threshold.txt','w') as file:
            file.write(best_threshold)
        # print_results(y_test,final_preds)

        # print("-----------------------------------------------------------------------------")
        # print("After computing best threshold")
        # print("-----------------------------------------------------------------------------")

        # print(f"Best threshold = {best_threshold:.3f}")
        # print(f"Precision = {best_precision:.3f}, Recall = {best_recall:.3f}, F1 = {best_f1:.3f}")

        # print("-----------------------------------------------------------------------------")
        # print("Final prediction")
        # print("-----------------------------------------------------------------------------")
        final_preds = (ensemble_proba >= best_threshold).astype(int)
        print_results(y_test, final_preds)

    else:
        raise ValueError

strategy = EnsembleStrategy(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
    )

try:
    models = load('model_v1.0.joblib')  # load
    while True:
        user_opinion=input("Trained model already exists. Do you wish to train a new model?(y/n)").lower()
        if user_opinion=='n':
            if models:
                preds = np.array([model.predict(X_test) for model in models])
                final_preds = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=preds)
                probas = np.array([model.predict_proba(X_test)[:, 1]
                       for model in models])
                ensemble_proba = probas.mean(axis=0)

            # 3) Compute precision–recall curve
                precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, ensemble_proba)

            # 4) (Optional) Plot it
                f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)

            # Find the threshold with highest F1
                best_idx = f1_scores.argmax()
                best_threshold = thresholds[best_idx]
                best_precision = precision_vals[best_idx]
                best_recall = recall_vals[best_idx]
                best_f1 = f1_scores[best_idx]
                final_preds = (ensemble_proba >= best_threshold).astype(int)
                with open('threshold.txt','w') as file:
                    file.write(str(best_threshold))
                print_results(y_test, final_preds)
                break
            else:
                FL_training()
                load_model_and_give_output(X_test)
                break
        elif user_opinion=='y':
            FL_training()
            load_model_and_give_output(X_test)
            break
        else:
            print("Kindly enter a valid prompt")
    
except FileNotFoundError:
    FL_training()
    load_model_and_give_output(X_test)
    
    

