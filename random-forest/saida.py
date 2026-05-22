# saida.py — geração do arquivo de saída do teste cego

import numpy as np
import pandas as pd
from config import ARQUIVO_SAIDA


def gerar_saida(ids, gravidades, classes):
    """Salva o arquivo CSV com as predições do teste cego."""
    df_out = pd.DataFrame({
        'i':      ids,
        'gravid': np.round(gravidades, 4),
        'classe': classes.astype(int)
    })
    df_out.to_csv(ARQUIVO_SAIDA, index=False)
    print(f"\nArquivo de saída salvo em: {ARQUIVO_SAIDA}")
    print(df_out.head(10).to_string(index=False))
