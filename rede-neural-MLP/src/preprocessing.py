
#preprocessing.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import joblib, os

SEED = 42


def carregar_com_label(caminho: str) -> pd.DataFrame:
    
    df = pd.read_csv(caminho, header=None).apply(pd.to_numeric, errors="coerce").dropna()
    return pd.DataFrame({
        "qPA":       df.iloc[:, 3].values,
        "pulso":     df.iloc[:, 4].values,
        "resp":      df.iloc[:, 5].values,
        "gravidade": df.iloc[:, 6].values,
        "classe":    df.iloc[:, 7].astype(int).values,
    })


def carregar_sem_label(caminho: str) -> pd.DataFrame:
    
    df = pd.read_csv(caminho, header=None).apply(pd.to_numeric, errors="coerce").dropna()
    return pd.DataFrame({
        "i":     df.iloc[:, 0].astype(int).values,
        "qPA":   df.iloc[:, 3].values,
        "pulso": df.iloc[:, 4].values,
        "resp":  df.iloc[:, 5].values,
    })


def separar_features_targets(df: pd.DataFrame):
    X     = df[["qPA", "pulso", "resp"]].values
    y_reg = df["gravidade"].values
    y_clf = df["classe"].values.astype(int)
    return X, y_reg, y_clf


def dividir_dados(X, y_reg, y_clf, prop_teste=0.15, prop_val=0.15):

    X_tmp, X_te, yr_tmp, yr_te, yc_tmp, yc_te = train_test_split(
        X, y_reg, y_clf, test_size=prop_teste, random_state=SEED, stratify=y_clf)
    X_tr, X_val, yr_tr, yr_val, yc_tr, yc_val = train_test_split(
        X_tmp, yr_tmp, yc_tmp,
        test_size=prop_val / (1 - prop_teste), random_state=SEED, stratify=yc_tmp)
    print(f"[Split] Treino: {len(X_tr)} | Validação: {len(X_val)} | Teste: {len(X_te)}")
    return X_tr, X_val, X_te, yr_tr, yr_val, yr_te, yc_tr, yc_val, yc_te


def criar_normalizador(X_train: np.ndarray, caminho_salvar: str = None) -> MinMaxScaler:

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    if caminho_salvar:
        os.makedirs(os.path.dirname(caminho_salvar), exist_ok=True)
        joblib.dump(scaler, caminho_salvar)
        print(f"[Normalização] Scaler salvo em: {caminho_salvar}")
    return scaler


def normalizar(scaler: MinMaxScaler, *arrays):
    return tuple(scaler.transform(a) for a in arrays)


def carregar_normalizador(caminho: str) -> MinMaxScaler:
    return joblib.load(caminho)

