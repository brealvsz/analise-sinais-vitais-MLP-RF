ARQUIVO_COM_LABEL = '02_treino_sinais_vitais_com_label.txt'
ARQUIVO_SEM_LABEL = '01_treino_sinais_vitais_sem_label.txt'
ARQUIVO_SAIDA        = 'output_rf_blind_test.csv'

# Nomes das colunas
COLUNAS_ENTRADA      = ['s3', 's4', 's5']   # qPA, pulso, respiração (s1 e s2 excluídos)
COLUNA_GRAVIDADE     = 'g'
COLUNA_CLASSE        = 'y'

# Reprodutibilidade
SEED = 42

# Proporções de divisão
PROPORCAO_TESTE      = 0.15
PROPORCAO_VALIDACAO  = 0.15   # sobre o total original

NOMES_CLASSES = ['1-Crítico', '2-Instável', '3-Pot.Estável', '4-Estável']
