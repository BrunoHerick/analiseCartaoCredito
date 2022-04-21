import pandas as pd
from sklearn import naive_bayes
from sklearn.preprocessing import LabelEncoder
#from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

dados = pd.read_csv("/home/brunohelghast/PROFISSIONAL/PYTHON/SCIKIT_LEARN/previsaoAprovacaoCartaoCredito/application_record.csv")
dados = dados.drop(columns=["ID"])
dados.drop_duplicates(inplace=True)
dados = dados.dropna()

tabelaDados = dados.iloc[:,1:17].values
resultado = dados.iloc[:,17].values

for i in range(0,len(tabelaDados[5,:]),1):
    tabelaDados[:,i] = LabelEncoder().fit_transform(tabelaDados[:,i])

#naive_tabelaDados = GaussianNB()
arvore_tabelaDados = DecisionTreeClassifier(criterion="entropy")
arvore_tabelaDados.fit(tabelaDados, resultado)

print(arvore_tabelaDados.predict([[0, 1, 0, 80, 1, 1, 1, 1, 4304, 2830, 0, 0, 1, 1, 11, 1]]))
print(arvore_tabelaDados.predict([[0, 0, 0, 18, 0, 1, 2, 1, 975, 3483, 0, 0, 1, 0, 3, 0]]))
print(arvore_tabelaDados.predict([[1, 0, 0, 80, 0, 1, 2, 1, 975, 3483, 0, 0, 1, 0, 15, 1]]))

matris_corelacao = dados.corr()
print(matris_corelacao["CNT_FAM_MEMBERS"].sort_values(ascending=False))

# Build a machine learning model to predict if an applicant is 'good' or 'bad' client, different from other tasks, the definition of 'good' or 'bad' is not given. You should use some techique, such as vintage analysis to construct you label. Also, unbalance data problem is a big problem in this task. 
# 0: 1-29 dias de atraso 1: 30-59 dias de atraso 2: 60-89 dias de atraso 3: 90-119 dias de atraso 4: 120-149 dias de atraso 5: Dívidas vencidas ou incobráveis, baixas por mais de 150 dias C: quitado naquele mês X: Sem empréstimo no mês
# r² = 0.90 para CNT_FAM_MEMBERS e CNT_CHILDREN
# São considerados cliente boms com status igual a 0,C ou X