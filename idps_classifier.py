#IDPS classifier
#extraindo dados e gerando arquivos 


import random
import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Conjunto de aminoácidos desordenados
disordered_aminoacids = {'E', 'K', 'O', 'P', 'S'}

# Função para carregar sequências de um arquivo FASTA e rotulá-las
def sequences_upload(fasta_file, label):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append((str(record.seq), label))
    return sequences

# Caminhos para os arquivos FASTA
fasta_file_disordered = 'disordered.fasta'
fasta_file_ordered = 'ordered.fasta'

# Carregar sequências desordenadas (label=1) e não desordenadas (label=0)
disordered_sequences = sequences_upload(fasta_file_disordered, 1)
ordered_sequences = sequences_upload(fasta_file_ordered, 0)

# Combinar as sequências em um único DataFrame
sequences = disordered_sequences + ordered_sequences
df = pd.DataFrame(sequences, columns=['sequence', 'label'])

# Função para calcular o percentual de aminoácidos desordenados
def percentage_disorder(sequence):
    total_aminoacids = len(sequence)
    disordered = sum(1 for aa in sequence if aa in disordered_aminoacids)
    percentage = (disordered / total_aminoacids) * 100
    return percentage

# Adicionar o percentual de desordem ao DataFrame
df['percentage_disorder'] = df['sequence'].apply(percentage_disorder)

# Separar os recursos e os rótulos
X = df[['percentage_disorder']]
y = df['label']


#validação cruzada 
# Dividir o conjunto de dados em treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o classificador de Regressão Logística
clf = LogisticRegression(penalty='l2', C=1.0, random_state=42)

# Realizar a validação cruzada Leave-One-Out
loo = LeaveOneOut()
cv_scores = cross_val_score(clf, X_train, y_train, cv=loo, scoring='accuracy')

# Exibir os resultados da validação cruzada
#print(f"Leave-One-Out Accuracy Scores: {cv_scores}")
print(f"Mean LOO Accuracy: {np.mean(cv_scores):.2f}")


# Treinar o modelo com todos os dados de treinamento
clf.fit(X_train, y_train)

# Avaliar no conjunto de teste
y_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Função para prever desordem em uma nova sequência
def disorder_prediction(sequence):
    percentage = percentage_disorder(sequence)
    if percentage > 30:
        result = f"The protein is disordered: {percentage:.2f}% of disorder"
    else:
        result = f"The protein is structured: {percentage:.2f}% of disordered"
    return result


#testando o modelo 
# Geração de Sequências Aleatórias e Previsão

from Bio.Seq import Seq

# Conjunto de aminoácidos
aminoacids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Função para gerar uma sequência aleatória de aminoácidos
def random_sequence(size):
    return ''.join(random.choices(aminoacids, k=size))

# Gerar uma nova sequência aleatória e prever sua desordem
size_sequence = 100  # ou qualquer tamanho desejado
new_sequence = random_sequence(size_sequence)
result = disorder_prediction(new_sequence)
print(f"The new sequence is: {new_sequence}")
print(result)
