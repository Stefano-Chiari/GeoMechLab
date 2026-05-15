import numpy as np
import joblib
from pathlib import Path

def load_emulator(model_filepath):
    """
    Carica il GaussianProcessRegressor addestrato nel Checkpoint 2.
    """
    path = Path(model_filepath)
    if not path.exists():
        raise FileNotFoundError(f"Errore critico: Il file del modello non è presente al percorso {path}.")
    
    # joblib è lo standard de facto per modelli scikit-learn
    emulator = joblib.load(path)
    return emulator

def predict_displacement(emulator, layer_id, pressure, center_x, center_y):
    """
    Esegue l'inferenza usando il Processo Gaussiano.
    """
    # 1. Composizione del vettore. 
    # L'ordine DEVE essere rigorosamente quello del training del CP2.
    X_input = np.array([[layer_id, pressure, center_x, center_y]])
    
    # 2. Inferenza. 
    # Disabilitiamo il calcolo dell'intervallo di confidenza al 90% (return_std=False) 
    # per avere solo il profilo deterministico (mean) che serve all'ottimizzatore.
    u_pred_mean = emulator.predict(X_input, return_std=False)
    
    # 3. Pulizia della forma. 
    # Riduciamo eventuali dimensioni extra (es. [1, N] diventa [N])
    # per poter sottrarre agevolmente il vettore top_u_obs nel calcolo del Mean Squared Error.
    return np.squeeze(u_pred_mean)