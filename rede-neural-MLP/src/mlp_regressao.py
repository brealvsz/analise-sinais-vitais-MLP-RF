import numpy as np
import joblib
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

CONFIG_REGRESSAO = {
  
    "hidden_layer_sizes": (24, 12),
    "activation": "tanh",
    "solver": "adam",
    "learning_rate_init": 0.001,
    "max_iter": 2000,
    "early_stopping": True,      
    "random_state": 42,
    "tol": 1e-6
}

def criar_modelo_regressao() -> MLPRegressor:
    return MLPRegressor(**CONFIG_REGRESSAO)

def treinar_regressao(modelo: MLPRegressor, X_train: np.ndarray, y_train: np.ndarray):
    print(f"  Treinando Regressão: Input(3) → Hidden{CONFIG_REGRESSAO['hidden_layer_sizes']} → Output(1)")
    
    y_scaler = MinMaxScaler()
    y_train_n = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    modelo.fit(X_train, y_train_n)
    modelo._y_scaler = y_scaler  
    
   
    loss_rmse = [np.sqrt(v) * (y_scaler.data_max_[0] - y_scaler.data_min_[0]) for v in modelo.loss_curve_]
    
    print(f"  Épocas treinadas: {len(modelo.loss_curve_)}")
    return modelo, loss_rmse

def predizer_gravidade(modelo: MLPRegressor, X: np.ndarray) -> np.ndarray:
    
    return modelo._y_scaler.inverse_transform(modelo.predict(X).reshape(-1, 1)).ravel()

def salvar_modelo_regressao(modelo, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    joblib.dump(modelo, caminho)