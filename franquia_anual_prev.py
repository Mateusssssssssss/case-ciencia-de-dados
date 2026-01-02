import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

#Carregar o dataset
dados = pd.read_csv('slr12.csv',sep=';')
print(dados.head())\
#Descrição dos dados
print(dados.describe())
#Verificar valores nulos
dados_null = dados.isnull().sum()
print(f'Quantidade de dados null: {dados_null}')

#Verificar duplicatas
#Valor duplicado achado, após analise, identificou-se que não era,
#pois o valor True da duplicata era apenas em uma coluna, e não levava
#em consideração a o valor da segunda coluna.
#Logo não há dados duplicados
dados_dup = dados.duplicated().sum()
dados_dup1 = dados[['FrqAnual', 'CusInic']].duplicated()
print(f'Quantidade de dados duplicados: {dados_dup}')
print(dados_dup1)

#Variavel dependente y, variavel independente x
x = dados.iloc[:, 0]
y = dados.iloc[:, 1]
print(x)
# Valor da correlação entre x e y
correlacao = np.corrcoef(x, y)
print(f'Correlação entre x e y: {correlacao}')
#Correlação entre x e y = 0.47700725, logo Não é um bom modelo para previsões
#Pôs é um valor positivo fraco.

# formato de matriz com uma coluna a mais.
x = x.values.reshape(-1, 1)
#Criação do modelo e treinamento
#fit inicia o treinamento 
modelo = LinearRegression()
modelo.fit(x,y)


# Geração do gráfico com os pontos reais e as previsões
plt.scatter(x, y)
plt.plot(x, modelo.predict(x), color = 'red')
plt.show()

#Visualização do coeficiente
#Interceptação
# onde os dados encontram o eixo y
print(f'Interceptação: {modelo.intercept_}')

#Inclinação
#o quanto a minha variavel y cresce conforme a x cresce
#angulo da reta
print(f'Coeficiente{modelo.coef_}')

# calculo de R^2, coeficiente de determinação
print(f'R^2: {modelo.score(x,y)}')

#Grafico para visualizar os Residuos
#Verifica se temos um bom modelo
visualizador = ResidualsPlot(modelo)
# entender o relacionamento entre os dados para gerar visualizações úteis
visualizador.fit(x,y)
#exibir a visualização 
visualizador.poof()





