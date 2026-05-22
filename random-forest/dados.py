import pandas as pd
from sklearn.model_selection import train_test_split
from config import (ARQUIVO_COM_LABEL, ARQUIVO_SEM_LABEL,
                    COLUNAS_ENTRADA, COLUNA_GRAVIDADE, COLUNA_CLASSE,
                    SEED, PROPORCAO_TESTE, PROPORCAO_VALIDACAO)


def carregar_dados():
    df = pd.read_csv(ARQUIVO_COM_LABEL, header=None,
                     names=['i', 's1', 's2', 's3', 's4', 's5', 'g', 'y'])

    X      = df[COLUNAS_ENTRADA].values
    y_reg  = df[COLUNA_GRAVIDADE].values
    y_clf  = df[COLUNA_CLASSE].values.astype(int)

    print(f"Dataset carregado: {len(df)} amostras")
    print(f"Distribuição de classes:\n{pd.Series(y_clf).value_counts().sort_index().to_string()}\n")
    return X, y_reg, y_clf


def dividir_dados(X, y_reg, y_clf):
    val_ratio = PROPORCAO_VALIDACAO / (1 - PROPORCAO_TESTE)

    X_temp, X_test, yr_temp, yr_test, yc_temp, yc_test = train_test_split(
        X, y_reg, y_clf,
        test_size=PROPORCAO_TESTE,
        random_state=SEED,
        stratify=y_clf
    )
    X_train, X_val, yr_train, yr_val, yc_train, yc_val = train_test_split(
        X_temp, yr_temp, yc_temp,
        test_size=val_ratio,
        random_state=SEED,
        stratify=yc_temp
    )

    print(f"Divisão: Treino={len(X_train)} | Validação={len(X_val)} | Teste={len(X_test)}\n")
    return X_train, X_val, X_test, yr_train, yr_val, yr_test, yc_train, yc_val, yc_test


def carregar_teste_cego():
    df = pd.read_csv(ARQUIVO_SEM_LABEL, header=None,
                     names=['i', 's1', 's2', 's3', 's4', 's5', 'g'])
    X_cego = df[COLUNAS_ENTRADA].values
    ids    = df['i'].astype(int).values
    return X_cego, ids
