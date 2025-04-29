# Importar las bibliotecas necesarias
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

# Cargar el conjunto de datos Iris
iris = load_iris()

X, y = iris.data, iris.target
print( X, y)
# Crear el clasificador del Árbol de Decisión
clf = tree.DecisionTreeClassifier()

# Entrenar el modelo con los datos
clf = clf.fit(X, y)

# Exportar el árbol de decisión en formato DOT para su visualización
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names,  
                                filled=True, rounded=True,  
                                special_characters=True)  

# Crear el gráfico con graphviz
graph = graphviz.Source(dot_data)

# Guardar el gráfico como un archivo PDF (opcional)
graph.render("iris_decision_tree")

# Mostrar el gráfico directamente
graph.view()