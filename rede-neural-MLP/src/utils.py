
#utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

matplotlib.use("Agg") 

NOMES_CLASSES = {1: "Crítico", 2: "Instável", 3: "Pot. Estável", 4: "Estável"}


# ─────────────────────────────────────────────
#  CURVAS DE APRENDIZADO
# ─────────────────────────────────────────────

def plotar_curva_regressao(historico_train: list, caminho_ou_val, caminho: str = None,
                           val_rmse: float = None):
    
    if isinstance(caminho_ou_val, list):
        historico_val_antigo = caminho_ou_val
        caminho_final = caminho
    else:
        historico_val_antigo = None
        caminho_final = caminho_ou_val

    epocas = range(1, len(historico_train) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epocas, historico_train, label="Treino (RMSE)", color="#2196F3", linewidth=1.8)

    if historico_val_antigo is not None:
        ax.plot(epocas, historico_val_antigo, label="Validação (RMSE)",
                color="#FF5722", linewidth=1.8, linestyle="--")
    elif val_rmse is not None:
        ax.axhline(float(val_rmse), color="#FF5722", linewidth=1.5, linestyle="--",
                   label=f"Validação final: {float(val_rmse):.4f}")

    ax.set_title("Curva de Aprendizado — MLP Regressão\n(RMSE por Época)", fontsize=13, pad=12)
    ax.set_xlabel("Época")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _salvar_figura(fig, caminho_final)


def plotar_curva_classificacao(historico_loss: list, acc_val_final: float, caminho: str):
   
    epocas = range(1, len(historico_loss) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epocas, historico_loss, label="Treino (Cross-Entropy)", color="#FF9800", linewidth=1.8)
    ax.axhline(acc_val_final, color="#9C27B0", linewidth=1.5, linestyle="--",
               label=f"Acurácia val. final: {acc_val_final:.4f}")
    ax.set_title("Curva de Aprendizado — MLP Classificação\n(Loss por Época)", fontsize=13, pad=12)
    ax.set_xlabel("Época")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _salvar_figura(fig, caminho)


# ─────────────────────────────────────────────
#  MATRIZ DE CONFUSÃO (HEATMAP)
# ─────────────────────────────────────────────

def plotar_matriz_confusao(matriz: np.ndarray,
                            caminho: str,
                            fase: str = "Teste"):
    
    nomes = [NOMES_CLASSES[c] for c in [1, 2, 3, 4]]
    total_real = matriz.sum(axis=1, keepdims=True)
    matriz_pct = np.where(total_real > 0, matriz / total_real * 100, 0)

    rotulos = np.array([
        [f"{v}\n({p:.1f}%)" for v, p in zip(linha_v, linha_p)]
        for linha_v, linha_p in zip(matriz, matriz_pct)
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matriz_pct, annot=rotulos, fmt="", cmap="Blues",
        xticklabels=nomes, yticklabels=nomes,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "% por classe real"}
    )
    ax.set_title(f"Matriz de Confusão — MLP Classificação ({fase})", fontsize=13, pad=12)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    fig.tight_layout()

    _salvar_figura(fig, caminho)


# ─────────────────────────────────────────────
#  PREDITO vs REAL (REGRESSÃO)
# ─────────────────────────────────────────────

def plotar_scatter_regressao(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              caminho: str,
                              fase: str = "Teste"):

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color="#2196F3", label="Amostras")

    lim_min = min(y_true.min(), y_pred.min()) - 1
    lim_max = max(y_true.max(), y_pred.max()) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.5, label="Predição perfeita")

    ax.set_title(f"Predito vs. Real — MLP Regressão ({fase})", fontsize=13)
    ax.set_xlabel("Gravidade Real")
    ax.set_ylabel("Gravidade Predita")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _salvar_figura(fig, caminho)


# ─────────────────────────────────────────────
#  ARQUIVO DE SAÍDA — TESTE CEGO
# ─────────────────────────────────────────────

def gerar_csv_predicoes(indices: np.ndarray,
                         gravidades_pred: np.ndarray,
                         classes_pred: np.ndarray,
                         caminho: str):
    df = pd.DataFrame({
        "i":         indices.astype(int),
        "gravidade": np.round(gravidades_pred, 4),
        "classe":    classes_pred.astype(int),
    })

    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    df.to_csv(caminho, index=False)
    print(f"\n  [Saída] Predições do teste cego salvas em: {caminho}")
    print(df.head(10).to_string(index=False))
    return df


def plotar_metricas_regressao(res_dict: dict, caminho: str):
    fases = list(res_dict.keys())
    rmse_vals = [res_dict[f]["rmse"] for f in fases]
    mae_vals = [res_dict[f]["mae"] for f in fases]
    r2_vals = [res_dict[f]["r2"] for f in fases]
    
    x = np.arange(len(fases))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, rmse_vals, width, label='RMSE', color="#2196F3")
    ax.bar(x, mae_vals, width, label='MAE', color="#4CAF50")
    ax.bar(x + width, r2_vals, width, label='R²', color="#FF9800")
    
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in fases])
    ax.legend()
    ax.set_title("Métricas de Regressão por Fase", fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _salvar_figura(fig, caminho)

def plotar_metricas_classificacao(res_dict: dict, caminho: str):
    fases = list(res_dict.keys())
    acc_vals = [res_dict[f]["acuracia"] for f in fases]
    prec_vals = [res_dict[f]["precision"] for f in fases]
    rec_vals = [res_dict[f]["recall"] for f in fases]
    f1_vals = [res_dict[f]["f1_mac"] for f in fases]
    
    x = np.arange(len(fases))
    width = 0.2  
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5*width, acc_vals, width, label='Acurácia', color="#9C27B0")
    ax.bar(x - 0.5*width, prec_vals, width, label='Precision', color="#3F51B5")
    ax.bar(x + 0.5*width, rec_vals, width, label='Recall', color="#FF9800")
    ax.bar(x + 1.5*width, f1_vals, width, label='F1-Macro', color="#00BCD4")
    
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in fases])
    ax.legend(loc='lower right') 
    ax.set_title("Métricas de Classificação por Fase", fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _salvar_figura(fig, caminho)

# ─────────────────────────────────────────────
#  UTILITÁRIO INTERNO
# ─────────────────────────────────────────────

def _salvar_figura(fig: plt.Figure, caminho: str):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Gráfico] Salvo em: {caminho}")