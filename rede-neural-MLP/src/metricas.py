import numpy as np
import os
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, mean_absolute_error, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

NOMES_CLASSES = {1: "Crítico", 2: "Instável", 3: "Pot. Estável", 4: "Estável"}

# ─────────────────────────────────────────────
#  REGRESSÃO
# ─────────────────────────────────────────────

def avaliar_regressao(modelo, X, y_true, fase):
    y_pred = modelo._y_scaler.inverse_transform(modelo.predict(X).reshape(-1, 1)).ravel()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"  [{fase}] Regressão -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "y_pred": y_pred, "y_true": y_true}

# ─────────────────────────────────────────────
#  CLASSIFICAÇÃO
# ─────────────────────────────────────────────

def avaliar_classificacao(modelo, X, y_true, fase, caminho_salvar=None):
    y_pred = modelo.predict(X)
    classes, nomes = [1, 2, 3, 4], [NOMES_CLASSES[c] for c in [1, 2, 3, 4]]
    
    acc = accuracy_score(y_true, y_pred)
    prec_mac = precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    rec_mac = recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    f1_mac = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    matriz = confusion_matrix(y_true, y_pred, labels=classes)
    
    print(f"  [{fase}] Classificação -> Acurácia: {acc:.4f} | Precision: {prec_mac:.4f} | Recall: {rec_mac:.4f} | F1-Macro: {f1_mac:.4f}")
    
    if caminho_salvar and fase == "Teste":
        relatorio = classification_report(y_true, y_pred, labels=classes, target_names=nomes, zero_division=0)
        texto = f"RELATÓRIO CLASSIFICAÇÃO ({fase})\n"
        texto += f"Acurácia: {acc:.4f}\nPrecision (Macro): {prec_mac:.4f}\nRecall (Macro): {rec_mac:.4f}\nF1-Score (Macro): {f1_mac:.4f}\n\n"
        texto += f"{relatorio}\nMatriz de Confusão:\n{matriz}"
        
        os.makedirs(os.path.dirname(caminho_salvar), exist_ok=True)
        with open(caminho_salvar, "w", encoding="utf-8") as f:
            f.write(texto)
            
    return {
        "acuracia": acc, 
        "precision": prec_mac, 
        "recall": rec_mac, 
        "f1_mac": f1_mac, 
        "matriz": matriz, 
        "y_pred": y_pred
    }