import pygame
import random
import pandas as pd
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils import resample



# Ignorar todos los warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

pygame.init()

w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Phaser remastered ultra pro 3000 no DLC incluido")

BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

jugador = None
bala_h = None
bala_v = None
fondo = None
nave = None
menu = None

salto = False
salto_altura = 15
gravedad = 1
en_suelo = True

posicion_original = 0
regreso_activo = False
tiempo_mov_derecha = 0

pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False
datos_modelo = []

jugador_frames = [
    pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/sprites/mono_frame_1.png'),
    pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/sprites/mono_frame_2.png'),
    pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/sprites/mono_frame_3.png'),
    pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/sprites/mono_frame_4.png')
]

bala_img = pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/sprites/purple_ball.png')
fondo_img = pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/game/fondo2.png')
nave_img = pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/game/ufo.png')
fondo_img = pygame.transform.scale(fondo_img, (w, h))

jugador = pygame.Rect(50, h - 100, 32, 48)
bala_h = pygame.Rect(w - 50, h - 90, 16, 16)
bala_v = pygame.Rect(random.randint(0, w - 16), 0, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)

current_frame = 0
frame_speed = 10
frame_count = 0

velocidad_bala_h = -10
bala_h_disparada = False
bala_v_disparada = False
velocidad_bala_v = 5

fondo_x1 = 0
fondo_x2 = w

modelo = None
modelo_actual = "arbol"  # Opciones: 'arbol', 'red', 'knn'

def reiniciar_csv():
    global datos_modelo
    datos_modelo = []
    columnas = [
        "velocidad_balax", "bala_h_x",
        "velocidad_balay", "bala_v_y",
        "accion_horizontal", "salto"
    ]
    df_vacio = pd.DataFrame(columns=columnas)
    ruta_csv = "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/datos_entrenamiento.csv"
    df_vacio.to_csv(ruta_csv, index=False)
    print("🧹 CSV reiniciado.")

def manejar_movimiento_auto():
    global jugador, salto, salto_altura, en_suelo, modelo_actual
    global posicion_original, regreso_activo, tiempo_mov_derecha, modo_auto

    if not modo_auto:
        return

    # ---------- SALTO ----------
    if bala_h_disparada:
        X_salto = pd.DataFrame([[velocidad_bala_h, bala_h.x - jugador.x]],
                               columns=["velocidad_balax", "bala_h_x"])

        try:
            if modelo_actual == "arbol":
                modelo_salto = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/decision_tree_h.pkl")
                accion_salto = modelo_salto.predict(X_salto)[0]
            elif modelo_actual == "knn":
                modelo_salto = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/k_neighbors_h.pkl")
                accion_salto = modelo_salto.predict(X_salto)[0]
            elif modelo_actual == "red":
                modelo_salto = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/red_neuronal_salto.pkl")
                scaler_salto = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/scaler_salto.pkl")
                X_salto_scaled = scaler_salto.transform(X_salto)
                accion_salto = modelo_salto.predict(X_salto_scaled)[0]
            else:
                accion_salto = 0
        except Exception as e:
            print("❌ Error al predecir salto:", e)
            accion_salto = 0

        if accion_salto == 1 and en_suelo and not salto:
            salto = True
            en_suelo = False

    # ---------- MOVIMIENTO HORIZONTAL ----------
    if bala_v_disparada and not regreso_activo:
        X_mov = pd.DataFrame([[velocidad_bala_h, bala_h.x - jugador.x, velocidad_bala_v, bala_v.y]],
                            columns=["velocidad_balax", "bala_h_x", "velocidad_balay", "bala_v_y"])

        try:
            if modelo_actual == "arbol":
                modelo_mov = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/decision_tree_v.pkl")
                accion_mov = modelo_mov.predict(X_mov)[0]
            elif modelo_actual == "knn":
                modelo_mov = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/k_neighbors_v.pkl")
                accion_mov = modelo_mov.predict(X_mov)[0]
            elif modelo_actual == "red":
                modelo_mov = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/red_neuronal_movimiento.pkl")
                scaler_mov = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/scaler_movimiento.pkl")
                X_mov_scaled = scaler_mov.transform(X_mov)
                accion_mov = modelo_mov.predict(X_mov_scaled)[0]
                print(f"📡 Movimiento predicho: {accion_mov} | Entrada: {X_mov.values.tolist()}")
            else:
                accion_mov = 0
        except Exception as e:
            print("❌ Error al predecir movimiento:", e)
            accion_mov = 0

        # Si se mueve a la derecha, guardar posición original y activar regreso
        if accion_mov == 1:
            posicion_original = jugador.x
            jugador.x += 25
            jugador.x = max(0, min(w - jugador.width, jugador.x))
            regreso_activo = True
            tiempo_mov_derecha = pygame.time.get_ticks()

        elif accion_mov == -1:
            jugador.x -= 5
            jugador.x = max(0, min(w - jugador.width, jugador.x))

    # ---------- REGRESO AUTOMÁTICO ----------
    if regreso_activo:
        tiempo_actual = pygame.time.get_ticks()
        if tiempo_actual - tiempo_mov_derecha >= 1000:  # después de 1 segundo
            jugador.x = posicion_original
            jugador.x = max(0, min(w - jugador.width, jugador.x))
            regreso_activo = False

# ... ENTRENAMIENTO - K NEIGHBORS ...
def entrenar_k_neighbors():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.utils import resample
    from joblib import dump
    import pandas as pd

    ruta_dataset = "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/datos_entrenamiento.csv"
    ruta_modelos = "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/"

    try:
        df = pd.read_csv(ruta_dataset)

        if df.empty:
            print("⚠️ Dataset vacío. No se puede entrenar.")
            return

        # -----------------------------------
        # 🧠 MODELO PARA SALTO
        # -----------------------------------
        if "salto" in df.columns:
            X_salto = df[["velocidad_balax", "bala_h_x"]]
            y_salto = df["salto"]

            if y_salto.nunique() < 2:
                print("⚠️ No hay suficientes clases para entrenar el modelo de salto.")
            else:
                X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                    X_salto, y_salto, test_size=0.2, stratify=y_salto, random_state=42
                )

                modelo_salto = KNeighborsClassifier(n_neighbors=6)
                modelo_salto.fit(X_train_s, y_train_s)

                print("\n🔍 Evaluación del modelo KNN (salto):")
                print(classification_report(y_test_s, modelo_salto.predict(X_test_s)))

                dump(modelo_salto, ruta_modelos + "k_neighbors_h.pkl")
        else:
            print("⚠️ Columna 'salto' no encontrada en el dataset.")

        # -----------------------------------
        # 🧠 MODELO PARA MOVIMIENTO HORIZONTAL (balanceado)
        # -----------------------------------
        if "accion_horizontal" in df.columns:
            df_mov = df[df["accion_horizontal"].isin([0, 1])]
            df_0 = df_mov[df_mov["accion_horizontal"] == 0]
            df_1 = df_mov[df_mov["accion_horizontal"] == 1]

            if len(df_1) == 0 or len(df_0) == 0:
                print("⚠️ No hay suficientes muestras para balancear movimiento horizontal.")
            else:
                df_1_up = resample(df_1, replace=True, n_samples=len(df_0), random_state=42)
                df_bal = pd.concat([df_0, df_1_up])

                X_mov = df_bal[["velocidad_balax", "bala_h_x", "velocidad_balay", "bala_v_y"]]
                y_mov = df_bal["accion_horizontal"]

                X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
                    X_mov, y_mov, test_size=0.2, stratify=y_mov, random_state=42
                )

                modelo_mov = KNeighborsClassifier(n_neighbors=3)
                modelo_mov.fit(X_train_m, y_train_m)

                print("\n🔍 Evaluación del modelo KNN (movimiento horizontal):")
                print(classification_report(y_test_m, modelo_mov.predict(X_test_m)))

                dump(modelo_mov, ruta_modelos + "k_neighbors_v.pkl")
        else:
            print("⚠️ Columna 'accion_horizontal' no encontrada en el dataset.")

        print("\n✅ Modelos KNN entrenados y guardados correctamente.")

    except Exception as e:
        print(f"❌ Error durante el entrenamiento de KNN: {e}")

# ... ENTRENAMIENTO - DECISION TREE ...
def entrenar_arboles():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.utils import resample
    from joblib import dump
    import pandas as pd

    ruta_dataset = "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/datos_entrenamiento.csv"
    ruta_modelos = "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/"

    try:
        df = pd.read_csv(ruta_dataset)

        if df.empty:
            print("⚠️ Dataset vacío. No se puede entrenar.")
            return

        # -----------------------------------
        # 🌳 MODELO PARA SALTO
        # -----------------------------------
        if "salto" in df.columns:
            X_salto = df[["velocidad_balax", "bala_h_x"]]
            y_salto = df["salto"]

            if y_salto.nunique() < 2:
                print("⚠️ No hay suficientes clases para entrenar el modelo de salto.")
            else:
                X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                    X_salto, y_salto, test_size=0.2, stratify=y_salto, random_state=42
                )

                modelo_salto = DecisionTreeClassifier(max_depth=5, random_state=42)
                modelo_salto.fit(X_train_s, y_train_s)

                print("\n🔍 Evaluación del árbol de decisión (salto):")
                print(classification_report(y_test_s, modelo_salto.predict(X_test_s)))

                dump(modelo_salto, ruta_modelos + "decision_tree_h.pkl")
        else:
            print("⚠️ Columna 'salto' no encontrada en el dataset.")

        # -----------------------------------
        # 🌳 MODELO PARA MOVIMIENTO HORIZONTAL (balanceado)
        # -----------------------------------
        if "accion_horizontal" in df.columns:
            df_mov = df[df["accion_horizontal"].isin([0, 1])]
            df_0 = df_mov[df_mov["accion_horizontal"] == 0]
            df_1 = df_mov[df_mov["accion_horizontal"] == 1]

            if len(df_1) == 0 or len(df_0) == 0:
                print("⚠️ No hay suficientes muestras para balancear movimiento horizontal.")
            else:
                df_1_up = resample(df_1, replace=True, n_samples=len(df_0), random_state=42)
                df_bal = pd.concat([df_0, df_1_up])

                X_mov = df_bal[["velocidad_balax", "bala_h_x", "velocidad_balay", "bala_v_y"]]
                y_mov = df_bal["accion_horizontal"]

                X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
                    X_mov, y_mov, test_size=0.2, stratify=y_mov, random_state=42
                )

                modelo_mov = DecisionTreeClassifier(max_depth=5, random_state=42)
                modelo_mov.fit(X_train_m, y_train_m)

                print("\n🔍 Evaluación del árbol de decisión (movimiento horizontal):")
                print(classification_report(y_test_m, modelo_mov.predict(X_test_m)))

                dump(modelo_mov, ruta_modelos + "decision_tree_v.pkl")
        else:
            print("⚠️ Columna 'accion_horizontal' no encontrada en el dataset.")

        print("\n✅ Modelos de árbol entrenados y guardados correctamente.")

    except Exception as e:
        print(f"❌ Error durante el entrenamiento de árboles de decisión: {e}")

# ... ENTRENAMIENTO - RED NEURONAL CON FUNCIÓN DE ACTIVACIÓN TANH ...
def entrenar_red_neuronal():
    ruta_dataset = "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/datos_entrenamiento.csv"
    ruta_modelos = "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/"

    try:
        df = pd.read_csv(ruta_dataset)

        if df.empty:
            print("⚠️ Dataset vacío. No se puede entrenar.")
            return

        # -----------------------------------
        # 🧠 MODELO PARA SALTO
        # -----------------------------------
        if "salto" in df.columns:
            X_salto = df[["velocidad_balax", "bala_h_x"]]
            y_salto = df["salto"]

            if y_salto.nunique() < 2:
                print("⚠️ No hay suficientes clases para entrenar el modelo de salto.")
            else:
                X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                    X_salto, y_salto, test_size=0.2, stratify=y_salto, random_state=42
                )

                scaler_salto = StandardScaler()
                X_train_s_scaled = scaler_salto.fit_transform(X_train_s)
                X_test_s_scaled = scaler_salto.transform(X_test_s)

                modelo_salto = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=500, random_state=42)
                modelo_salto.fit(X_train_s_scaled, y_train_s)

                print("\n🔍 Evaluación del modelo de salto:")
                print(classification_report(y_test_s, modelo_salto.predict(X_test_s_scaled)))

                dump(modelo_salto, ruta_modelos + "red_neuronal_salto.pkl")
                dump(scaler_salto, ruta_modelos + "scaler_salto.pkl")
        else:
            print("⚠️ Columna 'salto' no encontrada en el dataset.")

        # -----------------------------------
        # 🧠 MODELO PARA MOVIMIENTO HORIZONTAL (balanceado)
        # -----------------------------------
        if "accion_horizontal" in df.columns:
            df_mov = df[df["accion_horizontal"].isin([0, 1])]
            df_0 = df_mov[df_mov["accion_horizontal"] == 0]
            df_1 = df_mov[df_mov["accion_horizontal"] == 1]

            if len(df_1) == 0 or len(df_0) == 0:
                print("⚠️ No hay suficientes muestras para balancear movimiento horizontal.")
            else:
                df_1_up = resample(df_1, replace=True, n_samples=len(df_0), random_state=42)
                df_bal = pd.concat([df_0, df_1_up])

                X_mov = df_bal[["velocidad_balax", "bala_h_x", "velocidad_balay", "bala_v_y"]]
                y_mov = df_bal["accion_horizontal"]

                X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
                    X_mov, y_mov, test_size=0.2, stratify=y_mov, random_state=42
                )

                scaler_mov = StandardScaler()
                X_train_m_scaled = scaler_mov.fit_transform(X_train_m)
                X_test_m_scaled = scaler_mov.transform(X_test_m)

                modelo_mov = MLPClassifier(hidden_layer_sizes=(8, 4), activation='relu', max_iter=500, random_state=42)
                modelo_mov.fit(X_train_m_scaled, y_train_m)

                print("\n🔍 Evaluación del modelo de movimiento horizontal:")
                print(classification_report(y_test_m, modelo_mov.predict(X_test_m_scaled)))

                dump(modelo_mov, ruta_modelos + "red_neuronal_movimiento.pkl")
                dump(scaler_mov, ruta_modelos + "scaler_movimiento.pkl")
        else:
            print("⚠️ Columna 'accion_horizontal' no encontrada en el dataset.")

        print("\n✅ Proceso de entrenamiento finalizado.")

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")


def disparar_bala_h():
    global bala_h_disparada, velocidad_bala_h
    if not bala_h_disparada:
        velocidad_bala_h = random.randint(-8, -5)
        bala_h_disparada = True

def disparar_bala_v():
    global bala_v_disparada, velocidad_bala_v
    if not bala_v_disparada:
        velocidad_bala_v = 5 # random.randint(4, 5)
        bala_v.x = 50
        bala_v_disparada = True

def reset_bala_h():
    global bala_h, bala_h_disparada
    bala_h.x = w - 50
    bala_h_disparada = False

def reset_bala_v():
    global bala_v, bala_v_disparada
    bala_v.y = 0
    bala_v.x = 50
    bala_v_disparada = False

def guardar_csv():
    df = pd.DataFrame(datos_modelo, columns=[
        "velocidad_balax", "bala_h_x", "velocidad_balay", "bala_v_y", "accion_horizontal", "salto"
    ])
    df.to_csv("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/datos_entrenamiento.csv", index=False)

def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo
    if salto:
        jugador.y -= salto_altura
        salto_altura -= gravedad
        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 15
            en_suelo = True

def guardar_datos():
    global modo_auto
    if modo_auto:
        return  # No guardar datos si estamos en modo automático
    salto_hecho = 1 if salto else 0
    teclas = pygame.key.get_pressed()
    accion_horizontal = -1 if teclas[pygame.K_LEFT] else 1 if teclas[pygame.K_RIGHT] else 0
    fila = (
        velocidad_bala_h, bala_h.x - jugador.x,
        velocidad_bala_v, bala_v.y,
        accion_horizontal, salto_hecho
    )
    print("💾 Fila guardada:", fila)
    datos_modelo.append(fila)


def pausa_juego():
    global pausa, menu_activo, modo_auto, modelo_actual
    pausa = not pausa
    guardar_datos()
    guardar_csv()
    entrenar_red_neuronal()
    entrenar_arboles()
    entrenar_k_neighbors()

    if pausa:
        pantalla.fill(NEGRO)
        texto = fuente.render("PAUSA - Presiona 'A' para modo Auto o 'P' para reanudar", True, BLANCO)
        texto2 = fuente.render("Presiona '1': Árbol | '2': Red Neuronal | '3': K-Vecinos", True, BLANCO)
        pantalla.blit(texto, (w // 6, h // 2))
        pantalla.blit(texto2, (w // 6, h // 2 + 30))
        pygame.display.flip()

        while pausa:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    pygame.quit()
                    guardar_csv()
                    exit()

                elif evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_a:
                        modo_auto = True
                        pausa = False
                        print("🤖 Modo automático activado")

                    elif evento.key == pygame.K_1:
                        modelo_actual = "arbol"
                        entrenar_arboles()
                        print("🌳 Modelo seleccionado: 'arbol'")

                    elif evento.key == pygame.K_2:
                        modelo_actual = "red"
                        entrenar_red_neuronal()
                        print("🤖 Modelo seleccionado: 'red'")

                    elif evento.key == pygame.K_3:
                        modelo_actual = "knn"
                        entrenar_k_neighbors()
                        print("🤝 Modelo seleccionado: 'knn'")

                    elif evento.key == pygame.K_p:
                        pausa = False
                        print("▶️ Reanudando juego")

def mostrar_menu():
    global menu_activo, modo_auto, modelo_actual
    pantalla.fill(NEGRO)
    texto1 = fuente.render("Presiona 'A' para Auto, 'M' para Manual, 'Q' para Salir", True, BLANCO)
    texto2 = fuente.render("Presiona '1': Árbol | '2': Red Neuronal | '3': K-Vecinos", True, BLANCO)
    pantalla.blit(texto1, (w // 6, h // 2 - 30))
    pantalla.blit(texto2, (w // 6, h // 2 + 10))
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()

            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    modo_auto = True
                    menu_activo = False
                    print(f"🤖 Modo automático activado con modelo '{modelo_actual}'")

                elif evento.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                    reiniciar_csv()
                    print("Modo manual activado")

                elif evento.key == pygame.K_1:
                    modelo_actual = "arbol"
                    entrenar_arboles()
                    print("🤖 Modelo seleccionado: 'arbol'")

                elif evento.key == pygame.K_2:
                    modelo_actual = "red"
                    entrenar_red_neuronal()
                    print("🤖 Modelo seleccionado: 'red'")

                elif evento.key == pygame.K_3:
                    modelo_actual = "knn"
                    entrenar_k_neighbors()
                    print("🤖 Modelo seleccionado: 'knn'")

                elif evento.key == pygame.K_q:
                    guardar_csv()
                    pygame.quit()
                    exit()



def reiniciar_juego():
    global menu_activo, jugador, bala_h, bala_v, nave, bala_h_disparada, bala_v_disparada, salto, en_suelo
    menu_activo = True
    jugador.x, jugador.y = 50, h - 100
    reset_bala_h()
    reset_bala_v()
    nave.x, nave.y = w - 100, h - 100
    salto = False
    en_suelo = True
    mostrar_menu()

def update():
    global frame_count, current_frame, fondo_x1, fondo_x2

    fondo_x1 -= 1
    fondo_x2 -= 1
    if fondo_x1 <= -w: fondo_x1 = w
    if fondo_x2 <= -w: fondo_x2 = w
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0
    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))

    pantalla.blit(nave_img, (nave.x, nave.y))

    if bala_h_disparada:
        bala_h.x += velocidad_bala_h
    if bala_h.x < 0:
        reset_bala_h()
    pantalla.blit(bala_img, (bala_h.x, bala_h.y))

    if bala_v_disparada:
        bala_v.y += velocidad_bala_v
    if bala_v.y > h:
        reset_bala_v()
    pantalla.blit(bala_img, (bala_v.x, bala_v.y))

    if jugador.colliderect(bala_h) or jugador.colliderect(bala_v):
        guardar_csv()
        reiniciar_juego()

def main():
    global salto, en_suelo, bala_h_disparada, bala_v_disparada
    global modelo_salto, modelo_mov
    global mov_derecha_activado, tiempo_mov_derecha, posicion_anterior
    global posicion_original, regreso_activo, tiempo_mov_derecha

    mov_derecha_activado = False
    tiempo_mov_derecha = 0
    posicion_anterior = 0
    posicion_original = 0
    regreso_activo = False
    tiempo_mov_derecha = 0

    try:
        modelo_salto = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/k_neighbors_h.pkl")
        print("✅ Modelo de salto cargado.")
    except:
        print("❌ No se pudo cargar el modelo de salto.")

    try:
        modelo_mov = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/k_neighbors_v.pkl")
        print("✅ Modelo de movimiento cargado.")
    except:
        print("❌ No se pudo cargar el modelo de movimiento.")

    reloj = pygame.time.Clock()
    mostrar_menu()
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa and not modo_auto:
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:
                    pausa_juego()
                if evento.key == pygame.K_q:
                    guardar_csv()
                    pygame.quit()
                    exit()
                if evento.key == pygame.K_RIGHT and not modo_auto and not mov_derecha_activado:
                    posicion_anterior = jugador.x
                    jugador.x += 25
                    jugador.x = max(0, min(w - jugador.width, jugador.x))
                    mov_derecha_activado = True
                    tiempo_mov_derecha = pygame.time.get_ticks()

        # Regreso a la posición anterior tras 1s
        if mov_derecha_activado:
            tiempo_actual = pygame.time.get_ticks()
            if tiempo_actual - tiempo_mov_derecha >= 1000:
                jugador.x = posicion_anterior
                jugador.x = max(0, min(w - jugador.width, jugador.x))
                mov_derecha_activado = False

        if not pausa:
            teclas = pygame.key.get_pressed()

            if not bala_v_disparada:
                disparar_bala_v()
            if not bala_h_disparada:
                disparar_bala_h()

            if not modo_auto:
                if teclas[pygame.K_LEFT]:
                    jugador.x -= 5

            if modo_auto:
                manejar_movimiento_auto()

            if salto:
                manejar_salto()

            if not modo_auto:
                guardar_datos()

            update()

        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()