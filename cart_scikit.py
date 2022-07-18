from sklearn import tree
import graphviz 

training_data = [
    ['ensolarado', 50, 'sim'],
    ['ensolarado', 50, 'nao'],
    ['chuvoso',    10, 'sim'],
    ['chuvoso',    50, 'nao'],
    ['ensolarado', 10, 'sim'],
]

X = [[1, 50], [1, 50], [2, 10], [2, 50], [1, 10]]
Y = [1, 2, 1, 2, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
#tree.plot_tree(clf)

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=["tempo", "vento"], class_names=["Sim", "Não"], filled=True, rounded=True, special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 