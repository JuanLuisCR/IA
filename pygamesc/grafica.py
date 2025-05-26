# grafica.py
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Cargar datos
df = pd.read_csv("datos_entrenamiento.csv")

# ---------- GRÁFICO 3D de los datos ----------
fig = plt.figure(figsize=(10, 5))

# Subplot 1: datos en 3D
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Datos de entrenamiento (modo manual)")
ax1.set_xlabel("Velocidad")
ax1.set_ylabel("Distancia")
ax1.set_zlabel("Salto (target)")

# Separar por clase
clase_0 = df[df['salto'] == 0]
clase_1 = df[df['salto'] == 1]

ax1.scatter(clase_0["velocidad"], clase_0["distancia"], clase_0["salto"], c='blue', label='No saltó (0)', marker='o')
ax1.scatter(clase_1["velocidad"], clase_1["distancia"], clase_1["salto"], c='red', label='Saltó (1)', marker='^')
ax1.legend()

# ---------- COMPARACIÓN DE MODELOS ----------
# Preparar datos
X = df[["velocidad", "distancia"]]
y = df["salto"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Cargar modelos y evaluar
modelos = {
    "Árbol": joblib.load("modelo_arbol.pkl"),
    "Red Neuronal": joblib.load("modelo_nn.pkl"),
    "KNN": joblib.load("modelo_knn.pkl")
}

scores = {nombre: accuracy_score(y_test, modelo.predict(X_test)) for nombre, modelo in modelos.items()}

# Subplot 2: gráfica de barras
ax2 = fig.add_subplot(122)
ax2.set_title("Exactitud de modelos")
ax2.bar(scores.keys(), scores.values(), color=['green', 'blue', 'orange'])
ax2.set_ylim(0, 1)
ax2.set_ylabel("Accuracy")

# Mostrar todo
plt.tight_layout()
plt.show()