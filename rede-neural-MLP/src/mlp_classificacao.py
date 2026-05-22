import numpy as np
import joblib
import os
from sklearn.neural_network import MLPClassifier

CONFIG_CLASSIFICACAO = {
 
    "hidden_layer_sizes": (24, 16, 8), 
    "activation": "tanh",
    "solver": "adam",
    "learning_rate_init": 0.001,
    "max_iter": 2000,
    "early_stopping": False,     
    "random_state": 42,
    "tol": 1e-6
}

def criar_modelo_classificacao() -> MLPClassifier:
    return MLPClassifier(**CONFIG_CLASSIFICACAO)

def treinar_classificacao(modelo: MLPClassifier, X_train, y_train):
    print(f"  Treinando Classificação: Input(3) → Hidden{CONFIG_CLASSIFICACAO['hidden_layer_sizes']} → Softmax(4)")
    
    modelo.fit(X_train, y_train)
    print(f"  Épocas treinadas: {len(modelo.loss_curve_)}")
    
    return modelo, modelo.loss_curve_

def salvar_modelo_classificacao(modelo, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    joblib.dump(modelo, caminho)