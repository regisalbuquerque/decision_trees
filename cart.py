###############################################
# Métodos e Classes Utilitárias 
###############################################
""" Retorna os valores únicos de uma coluna do dataset."""
def unique_vals(rows, col):
    return set([row[col] for row in rows])

""" Retorna a quantidade de vezes que cada classe aparece no dataset."""
def class_counts(rows):  
    counts = {}  # Um dicionário do tipo classe -> quantidade.
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

"""Uma questão é usada para particionar um dataset.
A classe guarda o número da coluna e o valor da coluna. 
O método 'match' é usado para comparar o valor do atributo do exemplo
com o valor do atributo da questão
Se a coluna for númerica é feito a comparacão >=
Se a coluna for categórica é feito a comparacão ==
"""
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value
    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

"""Particiona um dataset de acordo com a questão.
Para cadas exemplo do dataset, é verificado se a condicão é verdadeira
ou falsa. Dividindo-se assim em dois grupos true_rows e false_rows
"""
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

###############################################
# Métricas
###############################################
"""Calcula o Indice GINI ou Impureza GINI de uma lista de exemplos.
https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
"""
def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

"""Cálculo do Ganho de Informacão (Information Gain).
É calculado pela incerteza do nó inicial, menos a impureza ponderada dos nós filhos.
"""
def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

###############################################
# Algoritmo Guloso 
###############################################
""" Encontra a melhor questão a ser perguntada (de divisão)
iterando por cada atributo / valor 
e calculando o ganho de informaćão de cada."""
def find_best_split(rows):
    best_gain = 0  
    best_question = None  
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  
    for col in range(n_features): 
        values = set([row[col] for row in rows])  
        for val in values:  
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

###############################################
# Componentes das Árvores
###############################################
"""
Um nó folha - Usado para classificar dados
É armazenado no nó folha um dicionário do tipo: classe -> quantidade
"""
class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)

"""Um nó de decisão
Guarda a questão e os nós filhos.
"""
class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

###############################################
# Treinamento do Modelo
###############################################
"""Constrói a árvore de forma recursiva.
"""
def build_tree(rows):
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)

###############################################
# Classificacão - Uso do Modelo
###############################################
def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

###############################################
# Impressão 
###############################################
def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return
    print (spacing + str(node.question))
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

###############################################
# Testes 
###############################################

training_data = [
    ['ensolarado', 50, 'sim'],
    ['ensolarado', 50, 'nao'],
    ['chuvoso',    10, 'sim'],
    ['chuvoso',    50, 'nao'],
    ['ensolarado', 10, 'sim'],
    
]

header = ["tempo", "vento", "joga"]

if __name__ == '__main__':

    my_tree = build_tree(training_data)
    print_tree(my_tree)

    # Evaluate
    testing_data = [
        ['chuvoso',    50, 'nao'],
        ['chuvoso',    10, 'sim'],
    ]

    for row in testing_data:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(classify(row, my_tree))))