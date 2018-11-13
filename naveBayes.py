import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import math
import string

##Leitura do CSV com os dados a serem estudados
text = pd.read_csv('teste.csv', sep=',' , names=['name','title','review','grade','rank'])
text.fillna('n', inplace=True) #Ajustando os espaços branco
train, test = train_test_split(text, test_size=0.1) #Usando 80% da base de dados
vetor_atributo = ['good', 'nice', 'great', 'bad', 'not good'] #vetor de sentimentos
#Utilizando os adjetivos presentes no tran.Review_Text segundo nltk
matrix = pd.Series.as_matrix(train)

vetor_frases = []
for i in range(len(matrix)):  # Criando o vetor com todas as frases
    auxx = ''
    for j in range(len(matrix[i])):
        if (isinstance(matrix[i][j], str)):
            if (len(matrix[i][j]) > len(auxx)):
                auxx = matrix[i][j]
    auxx = [char for char in auxx if char not in string.punctuation]
    auxx = ''.join(auxx).lower()
    vetor_frases = np.append(vetor_frases, auxx)

for i in range(len(vetor_frases)):
    frases = vetor_frases[i]
    if isinstance(frases, str):
        frases = [char for char in auxx if char not in string.punctuation]
        frases = ''.join(auxx).lower()
        words = set(frases.split())
        words = set(nltk.word_tokenize(frases))
        stopWord = set(stopwords.words('english'))
        stopWord = words - stopWord
        tagged = nltk.pos_tag(stopWord)
        for j in range(len(tagged)):
            if not (tagged[j][0] in vetor_atributo):
                vetor_atributo = np.append(vetor_atributo, tagged[j][0])

p1 = 0  # porcentagem de 1
p2 = 0  # porcentagem de 2
p3 = 0  # porcentagem de 3
error = []

v = []
for i in range(len(matrix)):  # Criando o vetor com os sentimentos de cada frase
    if (len(np.where(matrix[i] == '1')[0]) > 0 or len(np.where(matrix[i] == 1)[0]) > 0):
        v = np.append(v, int(1))
    elif (len(np.where(matrix[i] == '2')[0]) > 0 or len(np.where(matrix[i] == 2)[0]) > 0):
        v = np.append(v, int(2))
    elif (len(np.where(matrix[i] == '3')[0]) > 0 or len(np.where(matrix[i] == 3)[0]) > 0):
        v = np.append(v, int(3))
    else:
        error = np.append(error, i)

p1 = len(np.where(v == 1)[0])
p2 = len(np.where(v == 2)[0])
p3 = len(np.where(v == 3)[0])
aux = [p1, p2, p3]
p1 = p1 / np.sum(aux)
p2 = p2 / np.sum(aux)
p3 = p3 / np.sum(aux)
aux = [p1, p2, p3]

# matriz binária
matriz_binaria = []
for i in range(len(vetor_frases)):  # Criando uma matriz com vetores binarios em relação as
    vetor_aux = []
    if isinstance(vetor_frases[i], str) and not (len(np.where(error == i)[0])):
        for j in range(len(vetor_atributo)):
            if (vetor_atributo[j] in vetor_frases[i].lower()):
                vetor_aux = np.append(vetor_aux, int(1))
            else:
                vetor_aux = np.append(vetor_aux, int(0))
        if (i != 0):
            matriz_binaria = np.vstack([matriz_binaria, vetor_aux])
        else:
            matriz_binaria = vetor_aux
    else:
        error = np.append(error, i)

# Learning
vetor_resultado = np.zeros([3, len(vetor_atributo)])
maux = matriz_binaria.transpose()
for i in range(len(maux)):
    aux31 = 1  # Quandtidade de vezes que o atributo[i]=1 aparece quando o valor do sentimento=3
    aux11 = 1  # Quandtidade de vezes que o atributo[i]=1 aparece quando o valor do sentimento=1
    aux21 = 1  # Quandtidade de vezes que o atributo[i]=1 aparece quando o valor do sentimento=2
    perror = 0  # algumas frases podem não ser analisadas então é necessario ter a quantidade de erros acumulativos em cada frase
    for j in range(len(maux[0])):
        if (j in error):
            perror = perror + 1
        if (v[j + perror] == 1 and maux[i][j] == 1 and not (j in error)):
            aux11 = aux11 + 1
        if (v[j + perror] == 2 and maux[i][j] == 1 and not (j in error)):
            aux21 = aux21 + 1
        if (v[j + perror] == 3 and maux[i][j] == 1 and not (j in error)):
            aux31 = aux31 + 1
    vetor_resultado[0][i] = (aux11 / (2 + len(np.where(v == 1)[0])))
    vetor_resultado[1][i] = (aux21 / (2 + len(np.where(v == 2)[0])))
    vetor_resultado[2][i] = (aux31 / (2 + len(np.where(v == 3)[0])))

# test
nmatrix = pd.Series.as_matrix(test)
nvetor_frases = []
for i in range(len(nmatrix)):
    auxx = ''
    for j in range(len(nmatrix[i])):
        if (isinstance(nmatrix[i][j], str)):
            if (len(nmatrix[i][j]) > len(auxx)):
                auxx = nmatrix[i][j]
    auxx = re.sub(r"[^\w\s]", '', auxx)
    auxx = auxx.lower()
    nvetor_frases = np.append(nvetor_frases, auxx)
nv = []  # Vetor de sentimentos do data test
for i in range(len(nmatrix)):
    if (len(np.where(nmatrix[i] == '1')[0]) > 0 or len(np.where(nmatrix[i] == 1)[0]) > 0):
        nv = np.append(nv, int(1))
    elif (len(np.where(nmatrix[i] == '2')[0]) > 0 or len(np.where(nmatrix[i] == 2)[0]) > 0):
        nv = np.append(nv, int(2))
    elif (len(np.where(nmatrix[i] == '3')[0]) > 0 or len(np.where(nmatrix[i] == 3)[0]) > 0):
        nv = np.append(nv, int(3))
    else:
        error = np.append(error, i)

final = []  # Vetor binario dizendo se o algoritimo acertou
final_result = []  # Resultado calculado de cada probabilidade
fi = []
for z in range(len(nvetor_frases)):
    string_test = nvetor_frases[z]  # Texto para teste
    vetor_str_test = []  # V etor auxiliar binario da frase em relaçãoaos atributos
    result = 0
    test_result = np.zeros([3, 1])

    for j in range(len(vetor_atributo)):  # Criando o vetor_str_test
        if (vetor_atributo[j] in nvetor_frases[i].lower()):
            vetor_str_test = np.append(vetor_str_test, int(1))
        else:
            vetor_str_test = np.append(vetor_str_test, int(0))

    for i in range(len(vetor_resultado)):  # Fazendo o calculo da probabilidade de ser cada um dos sentimentos
        result = math.log(aux[i])
        for j in range(len(vetor_resultado[i])):
            result = result + (vetor_str_test[j]) * math.log(vetor_resultado[i][j]) + (
                        1 - vetor_str_test[j]) * math.log(1 - vetor_resultado[i][j])  # Modelo de Bernoulli
        test_result[i][0] = result

    aux4 = np.where(np.max(test_result) == test_result)[0] == (
                nv[z] - 1)  # Teste para ver se o valor encontrado pelo algoritimo é o mesmo do data test
    final = np.append(final, aux4[0])
    fi = np.append(fi, z)
    final_result = np.append(final_result, test_result)

raiting = len(np.where(final == True)[0]) / len(final)  # Taxa de acerto
final_result = np.reshape(final_result, [3, len(nvetor_frases)])

print(final_result)
print(raiting)