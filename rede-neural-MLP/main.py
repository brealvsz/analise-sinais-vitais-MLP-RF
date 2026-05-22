import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import (
    carregar_com_label, carregar_sem_label, separar_features_targets,
    dividir_dados, criar_normalizador, normalizar
)
from src.mlp_regressao import criar_modelo_regressao, treinar_regressao, predizer_gravidade, salvar_modelo_regressao
from src.mlp_classificacao import criar_modelo_classificacao, treinar_classificacao, salvar_modelo_classificacao
from src.metricas import avaliar_regressao, avaliar_classificacao
from src.utils import (
    plotar_curva_regressao, plotar_curva_classificacao,
    plotar_matriz_confusao, plotar_scatter_regressao, gerar_csv_predicoes,
    plotar_metricas_regressao, plotar_metricas_classificacao
)

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
CAMINHOS = {
    "train":             os.path.join(DIR_BASE, "data", "02_treino_sinais_vitais_com_label.csv"),
    "test_blind":        os.path.join(DIR_BASE, "data", "01_treino_sinais_vitais_sem_label.csv"),
    "scaler":            os.path.join(DIR_BASE, "models", "scaler.pkl"),
    "modelo_reg":        os.path.join(DIR_BASE, "models", "regressao", "mlp_regressao.pkl"),
    "modelo_clf":        os.path.join(DIR_BASE, "models", "classificacao", "mlp_classificacao.pkl"),
    "grafico_curva_reg": os.path.join(DIR_BASE, "outputs", "graficos", "curva_regressao.png"),
    "grafico_curva_clf": os.path.join(DIR_BASE, "outputs", "graficos", "curva_classificacao.png"),
    "grafico_matriz":    os.path.join(DIR_BASE, "outputs", "graficos", "matriz_confusao.png"),
    "grafico_scatter":   os.path.join(DIR_BASE, "outputs", "graficos", "scatter_regressao.png"),
    "grafico_barras_reg": os.path.join(DIR_BASE, "outputs", "graficos", "barras_regressao.png"),
    "grafico_barras_clf": os.path.join(DIR_BASE, "outputs", "graficos", "barras_classificacao.png"),
    "metricas_reg":      os.path.join(DIR_BASE, "outputs", "metricas", "relatorio_regressao.txt"),
    "metricas_clf":      os.path.join(DIR_BASE, "outputs", "metricas", "relatorio_classificacao.txt"),
    "predicoes":         os.path.join(DIR_BASE, "outputs", "predicoes", "predicoes_teste_cego.csv"),
}

def main():
    print("\n" + "█"*55)
    print("  Redes Neurais MLP: Regressão e Classificação")
    print("█"*55)

    #  1. PRÉ-PROCESSAMENTO 
    print("\n▶ ETAPA 1: PRÉ-PROCESSAMENTO")
    df_train = carregar_com_label(CAMINHOS["train"])
    X, y_reg, y_clf = separar_features_targets(df_train)

    X_tr, X_val, X_te, yr_tr, yr_val, yr_te, yc_tr, yc_val, yc_te = dividir_dados(X, y_reg, y_clf)
    scaler = criar_normalizador(X_tr, caminho_salvar=CAMINHOS["scaler"])
    X_tr_n, X_val_n, X_te_n = normalizar(scaler, X_tr, X_val, X_te)

    #  2. REGRESSÃO 
    print("\n▶ ETAPA 2: MLP REGRESSÃO")
    modelo_reg = criar_modelo_regressao()
    modelo_reg, hist_loss_reg = treinar_regressao(modelo_reg, X_tr_n, yr_tr)

    res_reg = {
        "treino": avaliar_regressao(modelo_reg, X_tr_n, yr_tr, "Treino"),
        "validacao": avaliar_regressao(modelo_reg, X_val_n, yr_val, "Validação"),
        "teste": avaliar_regressao(modelo_reg, X_te_n, yr_te, "Teste")
    }
    
    salvar_modelo_regressao(modelo_reg, CAMINHOS["modelo_reg"])
    plotar_curva_regressao(hist_loss_reg, CAMINHOS["grafico_curva_reg"])
    plotar_scatter_regressao(yr_te, res_reg["teste"]["y_pred"], CAMINHOS["grafico_scatter"])
    plotar_metricas_regressao(res_reg, CAMINHOS["grafico_barras_reg"])

    #  3. CLASSIFICAÇÃO
    print("\n▶ ETAPA 3: MLP CLASSIFICAÇÃO")
    modelo_clf = criar_modelo_classificacao()
    
    modelo_clf, hist_loss_clf = treinar_classificacao(modelo_clf, X_tr_n, yc_tr)

    res_clf = {
        "treino": avaliar_classificacao(modelo_clf, X_tr_n, yc_tr, "Treino", CAMINHOS["metricas_clf"]),
        "validacao": avaliar_classificacao(modelo_clf, X_val_n, yc_val, "Validação"),
        "teste": avaliar_classificacao(modelo_clf, X_te_n, yc_te, "Teste")
    }

    salvar_modelo_classificacao(modelo_clf, CAMINHOS["modelo_clf"])
    plotar_curva_classificacao(hist_loss_clf, res_clf["validacao"]["acuracia"], CAMINHOS["grafico_curva_clf"])
    plotar_matriz_confusao(res_clf["teste"]["matriz"], CAMINHOS["grafico_matriz"])
    plotar_metricas_classificacao(res_clf, CAMINHOS["grafico_barras_clf"])

    #  4. TESTE CEGO 
    print("\n▶ ETAPA 4: TESTE CEGO")
    if os.path.exists(CAMINHOS["test_blind"]):
        df_blind = carregar_sem_label(CAMINHOS["test_blind"])
        X_blind_n = scaler.transform(df_blind[["qPA", "pulso", "resp"]].values)

        grav_pred = predizer_gravidade(modelo_reg, X_blind_n)
        classe_pred = modelo_clf.predict(X_blind_n)

        gerar_csv_predicoes(df_blind["i"].values, grav_pred, classe_pred, CAMINHOS["predicoes"])
    else:
        print(f"  [Aviso] Arquivo não encontrado: {CAMINHOS['test_blind']}")

if __name__ == "__main__":
    main()