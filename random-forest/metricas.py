import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from config import NOMES_CLASSES


def avaliar_regressao(modelos_preds: dict):
    print("\n=== REGRESSÃO ===")
    print(f"{'Partição':<12} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 40)
    for nome, (y_real, y_pred) in modelos_preds.items():
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        mae  = mean_absolute_error(y_real, y_pred)
        r2   = r2_score(y_real, y_pred)
        print(f"{nome:<12} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f}")


def avaliar_classificacao(modelos_preds: dict):
    print("\n=== CLASSIFICAÇÃO ===")
    print(f"{'Partição':<12} {'Acurácia':>10} {'Precisão':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)
    for nome, (y_real, y_pred) in modelos_preds.items():
        acc  = accuracy_score(y_real, y_pred)
        prec = precision_score(y_real, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_real, y_pred, average='macro', zero_division=0)
        f1   = f1_score(y_real, y_pred, average='macro', zero_division=0)
        print(f"{nome:<12} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

    # Matriz de confusão e relatório detalhado só para o Teste
    y_real_test, y_pred_test = list(modelos_preds.values())[-1]
    print("\nMatriz de Confusão (Teste):")
    print(confusion_matrix(y_real_test, y_pred_test))
    print("\nRelatório Detalhado (Teste):")
    print(classification_report(y_real_test, y_pred_test,
                                target_names=NOMES_CLASSES,
                                zero_division=0))


def exibir_importancia_features(modelo):
    nomes = ['qPA (s3)', 'pulso (s4)', 'resp (s5)']
    print("Importância das Features:")
    for nome, imp in zip(nomes, modelo.feature_importances_):
        barra = '█' * int(imp * 40)
        print(f"  {nome:<12} {imp:.4f}  {barra}")
