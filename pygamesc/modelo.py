import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib

# Ruta a tu CSV local
df = pd.read_csv("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/datos_entrenamiento.csv")

# Entradas y salidas
X = df[["velocidad", "jugador_x", "bala_h_x", "bala_v_x"]]
y = df[["salto", "accion_horizontal"]]

# Entrenar
modelo = MultiOutputClassifier(DecisionTreeClassifier())
modelo.fit(X, y)

# Guardar
joblib.dump(modelo, "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/modelo_arbol_decision.pkl")
print("✅ Modelo entrenado y guardado con tu versión local de scikit-learn")