from dados    import carregar_dados, dividir_dados, carregar_teste_cego
from modelo   import treinar_regressao, treinar_classificacao
from metricas import avaliar_regressao, avaliar_classificacao, exibir_importancia_features
from saida    import gerar_saida


def main():
    print("=" * 55)
    print("  RANDOM FOREST — SINAIS VITAIS")
    print("=" * 55)

    X, y_reg, y_clf = carregar_dados()
    X_train, X_val, X_test, \
    yr_train, yr_val, yr_test, \
    yc_train, yc_val, yc_test = dividir_dados(X, y_reg, y_clf)

    print("Treinando modelo de Regressão...")
    rf_reg = treinar_regressao(X_train, yr_train)

    print("\nTreinando modelo de Classificação...")
    rf_clf = treinar_classificacao(X_train, yc_train)

    avaliar_regressao({
        'Treino':    (yr_train, rf_reg.predict(X_train)),
        'Validação': (yr_val,   rf_reg.predict(X_val)),
        'Teste':     (yr_test,  rf_reg.predict(X_test)),
    })

    avaliar_classificacao({
        'Treino':    (yc_train, rf_clf.predict(X_train)),
        'Validação': (yc_val,   rf_clf.predict(X_val)),
        'Teste':     (yc_test,  rf_clf.predict(X_test)),
    })

    print()
    exibir_importancia_features(rf_clf)

    X_cego, ids = carregar_teste_cego()
    gerar_saida(ids, rf_reg.predict(X_cego), rf_clf.predict(X_cego))

    print("\nConcluído!")


if __name__ == '__main__':
    main()
