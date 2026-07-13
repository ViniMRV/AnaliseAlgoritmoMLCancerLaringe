import pandas as pd
import os

nome_arquivo_entrada = 'dataset_classificacao_vogal_A_normal_Binario.csv'
nome_arquivo_saida = 'dataset_binario_saudavel_vs_patologico.csv'

df = pd.read_csv(nome_arquivo_entrada)

print("\n--- Contagem Original de Classes ---")
print("0 = Saudável | 1 = Benigno | 2 = Câncer")
print(df['Grupo_Alvo'].value_counts().sort_index())

df['Grupo_Binario'] = df['Grupo_Alvo'].apply(lambda x: 0 if x == 0 else 1)

df['Classe_Texto'] = df['Grupo_Binario'].map({0: 'Saudável', 1: 'Patológico'})

print("\n--- Nova Contagem (Binária) ---")
print("0 = Saudável | 1 = Patológico (Benigno + Câncer)")
print(df['Grupo_Binario'].value_counts().sort_index())

df.to_csv(nome_arquivo_saida, index=False)