# Avaliação Prática: Regressão Linear e Diagnóstico Estatístico
# Aluno: Lucas Pires de Castro Gradvohl
# Dataset: dataset_18.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

# Configuração de visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(font_scale=1.2)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Carrega o dataset
df = pd.read_csv('dataset_18.csv')

print("## Parte I - Análise Estatística ##")
print("\n1. Análise estatística inicial do conjunto de dados\n")

# Visualizando as primeiras linhas do dataset
print("Primeiras linhas do dataset:")
print(df.head())

# Informações gerais sobre o dataset
print("\nInformações gerais sobre o dataset:")
print(df.info())

# Verificando valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe().T)


# Distribuição da variável dependente
plt.figure(figsize=(10, 6))
sns.histplot(df['tempo_resposta'], kde=True)
plt.title('Distribuição do Tempo de Resposta')
plt.xlabel('Tempo de Resposta (ms)')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()

# Análise da relação entre variáveis numéricas e a variável dependente
numerical_cols = ['cpu_cores', 'ram_gb', 'latencia_ms', 'armazenamento_tb', 'tempo_resposta']
numerical_df = df[numerical_cols].copy()

# Matriz de correlação
plt.figure(figsize=(10, 8))
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação das Variáveis Numéricas')
plt.show()

# Análise das variáveis categóricas
categorical_cols = ['sistema_operacional', 'tipo_hd', 'tipo_processador']

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=col, y='tempo_resposta', data=df)
    plt.title(f'Tempo de Resposta por {col}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Estatísticas por categoria
    print(f"\nEstatísticas de tempo_resposta por {col}:")
    print(df.groupby(col)['tempo_resposta'].describe())

# Tratamento de dados faltantes
# Verificando a quantidade de dados faltantes
print("\nQuantidade de valores ausentes por coluna:")
missing_values = df.isnull().sum()
print(missing_values)

# Substituindo os valores ausentes nas variáveis numéricas pela mediana
for col in ['latencia_ms', 'armazenamento_tb']:
    if missing_values[col] > 0:
        df[col] = df[col].fillna(df[col].median())

# Para as variáveis categóricas, a moda
for col in categorical_cols:
    if missing_values[col] > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Verificando se ainda existem valores ausentes
print("\nValores ausentes após tratamento:")
print(df.isnull().sum())

print("\n## Parte II - Modelo e Diagnóstico ##")

# Preparando os dados para a regressão
# Tratando as variáveis categóricas usando variáveis dummy
df_model = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# Verificando as novas colunas dummy criadas
print("\nColunas após criação das variáveis dummy:")
print(df_model.columns.tolist())


# Identificando quais são as categorias base 
print("\nCategorias base (referência) para as variáveis categóricas:")
for col in categorical_cols:
    categories = df[col].unique()
    print(f"{col}: Base = {categories[0]}, outras categorias = {categories[1:]}")

# Preparando as variáveis para o modelo
X = df_model.drop('tempo_resposta', axis=1)
y = df_model['tempo_resposta']

X = X.astype(float)
y = y.astype(float)

# Adicionando uma constante para o intercepto
X = sm.add_constant(X)

# Ajustando o modelo de regressão linear múltipla
model = sm.OLS(y, X).fit()

# Exibindo o resumo do modelo
print("\nResumo do modelo de regressão linear múltipla:")
print(model.summary())

# Diagnóstico de multicolinearidade
# Calculando o VIF 
vif_data = pd.DataFrame()
vif_data["Variável"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nFator de Inflação da Variância (VIF):")
print(vif_data.sort_values("VIF", ascending=False))

# Diagnóstico de heterocedasticidade
# Visualizando os resíduos vs valores ajustados
fitted_values = model.fittedvalues
residuals = model.resid

plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Resíduos vs Valores Ajustados')
plt.xlabel('Valores Ajustados')
plt.ylabel('Resíduos')
plt.grid(True)
plt.show()

# Teste de Breusch-Pagan para heterocedasticidade
bp_test = het_breuschpagan(residuals, X)
print("\nTeste de Breusch-Pagan para heterocedasticidade:")
print(f"Estatística de teste: {bp_test[0]:.4f}")
print(f"p-valor: {bp_test[1]:.4f}")
print(f"Conclusão: {'Rejeita H0 - Presença de heterocedasticidade' if bp_test[1] < 0.05 else 'Não rejeita H0 - Não há evidência de heterocedasticidade'}")

print("\n## Parte III – Análise Crítica ##")

# Comparação de modelos
print("\nModelo 1: Todas as variáveis")
print(f"R² ajustado: {model.rsquared_adj:.4f}")
print(f"Estatística F: {model.fvalue:.4f}")
print(f"p-valor (F): {model.f_pvalue:.10f}")

# Identificando a variável com menor significância estatística ou com multicolinearidade
print("\nIdentificando variáveis para possível exclusão...")
p_values = model.pvalues.sort_values(ascending=False)
print("P-valores em ordem decrescente:")
print(p_values.head())

# Escolhendo uma variável para excluir com base no maior p-valor (menos significativa)
# ou maior VIF (multicolinearidade)
var_to_exclude = p_values.index[0]  # Variável com maior p-valor
print(f"\nVariável escolhida para exclusão: {var_to_exclude}")

# Criando o Modelo 2 sem a variável selecionada
X2 = X.drop(var_to_exclude, axis=1)
model2 = sm.OLS(y, X2).fit()

print("\nModelo 2: Sem a variável", var_to_exclude)
print(f"R² ajustado: {model2.rsquared_adj:.4f}")
print(f"Estatística F: {model2.fvalue:.4f}")
print(f"p-valor (F): {model2.f_pvalue:.10f}")

print("\nComparação entre os modelos:")
print(f"Diferença no R² ajustado: {model2.rsquared_adj - model.rsquared_adj:.6f}")

# Resumo do modelo final recomendado
print("\nResumo do modelo recomendado:")
print(model2.summary())




