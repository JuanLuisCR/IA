import pygame
import random
import pandas as pd
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
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

jugador = pygame.Rect(200, h - 100, 32, 48)
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
        "velocidad_balax", "bala_h_x", "tipo_bala_h",
        "velocidad_balay", "bala_v_y", "distancia_x",
        "accion_horizontal", "salto"
    ]
    df_vacio = pd.DataFrame(columns=columnas)
    ruta_csv = "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/datos_entrenamiento.csv"
    df_vacio.to_csv(ruta_csv, index=False)
    print("🧹 CSV reiniciado.")

def manejar_movimiento_auto():
    global jugador, salto, salto_altura, en_suelo, modelo_actual

    if not modo_auto:
        return

    # Decisión salto
    if bala_h_disparada:
        X_salto = pd.DataFrame([[velocidad_bala_h, bala_h.x - jugador.x, 0]], columns=["velocidad_balax", "bala_h_x", "tipo_bala_h"])

        try:
            if modelo_actual == "arbol":
                modelo_salto = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/decision_tree_h.pkl")
                if not hasattr(modelo_salto, "predict"):
                    raise ValueError("El modelo cargado no es válido.")
                accion_salto = modelo_salto.predict(X_salto)[0]
            elif modelo_actual == "knn":
                modelo_salto = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/k_neighbors_h.pkl")
                if not hasattr(modelo_salto, "predict"):
                    raise ValueError("El modelo cargado no es válido.")
                accion_salto = modelo_salto.predict(X_salto)[0]
            elif modelo_actual == "red":
                modelo_salto = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/red_neuronal_salto.pkl")
                accion_salto = modelo_salto.predict(X_salto)[0]
            else:
                accion_salto = 0
        except Exception as e:
            print("❌ Error al predecir salto:", e)
            accion_salto = 0

        if accion_salto == 1 and en_suelo and not salto:
            salto = True
            en_suelo = False

    # Decisión movimiento horizontal
        # Decisión movimiento horizontal
    if bala_v_disparada:
        distancia_x = bala_v.x - jugador.x  # Nueva distancia horizontal relevante
        X_mov = pd.DataFrame([[velocidad_bala_v, bala_v.y, distancia_x]],
            columns=["velocidad_balay", "bala_v_y", "distancia_x"])

        try:
            if modelo_actual == "arbol":
                modelo_mov = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/decision_tree_v.pkl")
                accion_mov = modelo_mov.predict(X_mov)[0]
            elif modelo_actual == "knn":
                modelo_mov = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/k_neighbors_v.pkl")
                accion_mov = modelo_mov.predict(X_mov)[0]
            elif modelo_actual == "red":
                modelo_mov = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/red_neuronal_movimiento.pkl")
                accion_mov = modelo_mov.predict(X_mov)[0]
            else:
                accion_mov = 0
        except Exception as e:
            print("❌ Error al predecir movimiento:", e)
            accion_mov = 0

        if accion_mov == -1:
            jugador.x -= 5
        elif accion_mov == 1:
            jugador.x += 5

        jugador.x = max(0, min(w - jugador.width, jugador.x))

# ... ENTRENAMIENTO - K NEIGHBORS ...
def entrenar_k_neighbors():
    try:
        df = pd.read_csv("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/datos_entrenamiento.csv")
        if len(df) < 10:
            print("⚠️ No hay suficientes datos para entrenar K-Neighbors.")
            return

        # Modelo de salto
        X_salto = df[["velocidad_balax", "bala_h_x", "tipo_bala_h"]]
        y_salto = df["salto"]

        modelo_salto = KNeighborsClassifier(n_neighbors=3)
        modelo_salto.fit(X_salto, y_salto)
        dump(modelo_salto, "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/k_neighbors_h.pkl")
        print("🤝 K-Neighbors para salto entrenado y guardado.")

        # Modelo de movimiento
        X_mov = df[["velocidad_balay", "bala_v_y", "distancia_x"]]
        y_mov = df["accion_horizontal"]

        modelo_mov = KNeighborsClassifier(n_neighbors=3)
        modelo_mov.fit(X_mov, y_mov)
        dump(modelo_mov, "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/k_neighbors_v.pkl")
        print("🤝 K-Neighbors para movimiento entrenado y guardado.")
        
    except Exception as e:
        print("❌ Error al entrenar K-Neighbors:", e)

# ... ENTRENAMIENTO - DECISION TREE ...
def entrenar_arboles():
    try:
        df = pd.read_csv("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/datos_entrenamiento.csv")
        if len(df) < 10:
            print("⚠️ No hay suficientes datos para entrenar árboles.")
            return

        # Modelo de salto con bala horizontal
        X_salto = df[["velocidad_balax", "bala_h_x", "tipo_bala_h"]]
        y_salto = df["salto"]

        modelo_salto = DecisionTreeClassifier()
        modelo_salto.fit(X_salto, y_salto)
        dump(modelo_salto, "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/decision_tree_h.pkl")
        print("🌳 Árbol de decisión para salto entrenado y guardado.")

        # Modelo de movimiento con bala vertical
        X_mov = df[["velocidad_balay", "bala_v_y", "distancia_x"]]
        y_mov = df["accion_horizontal"]

        modelo_mov = DecisionTreeClassifier()
        modelo_mov.fit(X_mov, y_mov)
        dump(modelo_mov, "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/decision_tree_v.pkl")
        print("🌳 Árbol de decisión para movimiento entrenado y guardado.")

    except Exception as e:
        print("❌ Error al entrenar árboles:", e)

# ... ENTRENAMIENTO - RED NEURONAL CON FUNCIÓN DE ACTIVACIÓN TANH ...

def entrenar_red_neuronal():
    try:
        df = pd.read_csv("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/datos_entrenamiento.csv")
        if len(df) < 10:
            print("⚠️ No hay suficientes datos para entrenar.")
            return

        # Red para salto
        X_salto = df[["velocidad_balax", "bala_h_x", "tipo_bala_h"]]
        y_salto = df["salto"]
        red_salto = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', max_iter=1000)
        red_salto.fit(X_salto, y_salto)
        dump(red_salto, "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/red_neuronal_salto.pkl")
        print("🤖 Red neuronal salto entrenada.")

        try:
            df = pd.read_csv("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/datos_entrenamiento.csv")

            df_0 = df[df["accion_horizontal"] == 0]
            df_1 = df[df["accion_horizontal"] == 1]
            df_m1 = df[df["accion_horizontal"] == -1]

            # Submuestreo de clase mayoritaria
            df_0_down = resample(df_0, replace=False, n_samples=800, random_state=42)
            # Sobremuestreo de clases minoritarias
            df_1_up = resample(df_1, replace=False, random_state=42)
            df_m1_up = resample(df_m1, replace=False, random_state=42)

            # Dataset balanceado
            df_bal = pd.concat([df_0_down, df_1_up, df_m1_up])

            X_mov = df_bal[["velocidad_balay", "bala_v_y", "distancia_x"]]
            y_mov = df_bal["accion_horizontal"]

            red_mov = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', max_iter=1500, solver='adam', random_state=42)
            red_mov.fit(X_mov, y_mov)
            dump(red_mov, "C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/model/red_neuronal_movimiento.pkl")
            print("✅ Red neuronal (balanceada) para movimiento horizontal entrenada correctamente.")

            print("Distribución de clases (accion_horizontal):")
            print(y_mov.value_counts())
        except Exception as e:
            print("❌ Error en entrenamiento balanceado de movimiento:", e)
        
    except Exception as e:
        print("❌ Error al entrenar redes neuronales:", e)



def disparar_bala_h():
    global bala_h_disparada, velocidad_bala_h
    if not bala_h_disparada:
        velocidad_bala_h = random.randint(-8, -3)
        bala_h_disparada = True
        guardar_datos(tipo_bala_h=0)    

def disparar_bala_v():
    global bala_v_disparada, velocidad_bala_v
    if not bala_v_disparada:
        velocidad_bala_v = random.randint(3, 5)
        bala_v_disparada = True

def reset_bala_h():
    global bala_h, bala_h_disparada
    bala_h.x = w - 50
    bala_h_disparada = False

def reset_bala_v():
    global bala_v, bala_v_disparada
    bala_v.y = 0
    bala_v.x = random.randint(0, w - 16)
    bala_v_disparada = False

def guardar_csv():
    df = pd.DataFrame(datos_modelo, columns=[
        "velocidad_balax", "bala_h_x", "tipo_bala_h", "velocidad_balay", "bala_v_y", "distancia_x", "accion_horizontal", "salto"
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

def guardar_datos(tipo_bala_h=0):
    global modo_auto
    if modo_auto:
        return  # No guardar datos si estamos en modo automático
    salto_hecho = 1 if salto else 0
    teclas = pygame.key.get_pressed()
    accion_horizontal = -1 if teclas[pygame.K_LEFT] else 1 if teclas[pygame.K_RIGHT] else 0
    fila = (
        velocidad_bala_h, bala_h.x - jugador.x, tipo_bala_h,
        velocidad_bala_v, bala_v.y, bala_v.x - jugador.x,
        accion_horizontal, salto_hecho
    )
    print("💾 Fila guardada:", fila)
    datos_modelo.append(fila)


def pausa_juego():
    global pausa, menu_activo, modo_auto, modelo_actual
    pausa = not pausa
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
    jugador.x, jugador.y = 250, h - 100
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
    global salto, en_suelo, bala_h_disparada, bala_v_disparada, modelo_salto, modelo_mov
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

        if not pausa:
            teclas = pygame.key.get_pressed()

            if not bala_v_disparada:
                disparar_bala_v()
            if not bala_h_disparada:
                disparar_bala_h()

            if not modo_auto:
                if teclas[pygame.K_LEFT]:
                    jugador.x -= 5
                if teclas[pygame.K_RIGHT]:
                    jugador.x += 5

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