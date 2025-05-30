import pygame
import random
#red neuronal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np


# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# Variables del jugador, bala, nave, fondo, etc.
jugador = None
bala = None
fondo = None
nave = None
menu = None

# Variables de salto
salto = False
salto_altura = 15  # Velocidad inicial de salto
gravedad = 1
en_suelo = True

#Variables de movimiento horizontal
mover_derecha = False
mover_izquierda = False
velocidad = 5  


# Variables de pausa y menú
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False  # Indica si el modo de juego es automático

# Lista para guardar los datos de velocidad, distancia y salto (target)
datos_modelo = []

#Red neuronal
modelo_nn = None
scaler_nn = None

# Cargar las imágenes
jugador_frames = [
    pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/sprites/mono_frame_1.png'),
    pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/sprites/mono_frame_2.png'),
    pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/sprites/mono_frame_3.png'),
    pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/sprites/mono_frame_4.png')
]

bala_img = pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/sprites/purple_ball.png')
fondo_img = pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/game/fondo2.png')
nave_img = pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/game/ufo.png')
menu_img = pygame.image.load('C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/assets/game/menu.png')

# Escalar la imagen de fondo para que coincida con el tamaño de la pantalla
fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear el rectángulo del jugador y de la bala
jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)  # Tamaño del menú

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0

# Variables para la bala
velocidad_bala = -30  # Velocidad de la bala hacia la izquierda
bala_disparada = False

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

#Segunda bala
bala_vertical = pygame.Rect(jugador.x, 0, 16, 16)
bala_vertical_disparada = False

# Función para disparar la bala
def disparar_bala():
    global bala_disparada, velocidad_bala, bala_vertical, bala_vertical_disparada
    if not bala_disparada:
        velocidad_bala = random.randint(-25, -10)  # Velocidad aleatoria negativa para la bala
        bala_disparada = True
    if not bala_vertical_disparada:
        bala_vertical.x = jugador.x  # Fijar posición X al jugador
        bala_vertical.y = 0
        bala_vertical_disparada = True

# Función para reiniciar la posición de la bala
def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50  # Reiniciar la posición de la bala
    bala_disparada = False

# Función para manejar el salto
def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura  # Mover al jugador hacia arriba
        salto_altura -= gravedad  # Aplicar gravedad (reduce la velocidad del salto)

        # Si el jugador llega al suelo, detener el salto
        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 15  # Restablecer la velocidad de salto
            en_suelo = True

# Función para actualizar el juego
def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2
    global bala_vertical, bala_vertical_disparada

    # Mover el fondo
    fondo_x1 -= 1
    fondo_x2 -= 1

    # Si el primer fondo sale de la pantalla, lo movemos detrás del segundo
    if fondo_x1 <= -w:
        fondo_x1 = w

    # Si el segundo fondo sale de la pantalla, lo movemos detrás del primero
    if fondo_x2 <= -w:
        fondo_x2 = w

    # Dibujar los fondos
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0

    # Dibujar el jugador con la animación
    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))
    #pygame.draw.rect(pantalla, (255, 0, 0), jugador, 2)  # Debug: dibuja el hitbox del jugador

    # Dibujar la nave
    pantalla.blit(nave_img, (nave.x, nave.y))

    # Bala horizontal
    if bala_disparada:
        bala.x += velocidad_bala
    if bala.x < 0:
        reset_bala()
    pantalla.blit(bala_img, (bala.x, bala.y))

    # Bala vertical
    if bala_vertical_disparada:
        bala_vertical.y += abs(10)  # Misma velocidad pero positiva
    if bala_vertical.y > h:
        bala_vertical.x = jugador.x  # Fijar nueva posición X del jugador
        bala_vertical.y = 0
    pantalla.blit(bala_img, (bala_vertical.x, bala_vertical.y))  # Reutilizamos la imagen

    # Colisiones
    if jugador.colliderect(bala) or jugador.colliderect(bala_vertical):
        print("Colisión detectada!")
        reiniciar_juego()

# Función para guardar datos del modelo en modo manual
def guardar_datos():
    global jugador, bala, velocidad_bala, salto
    distancia = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0  # 1 si saltó, 0 si no saltó
    # Guardar velocidad de la bala, distancia al jugador y si saltó o no
    datos_modelo.append((velocidad_bala, distancia, salto_hecho))

# Función para pausar el juego y guardar los datos
def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado. Datos registrados hasta ahora:", datos_modelo)
    else:
        print("Juego reanudado.")

# Función para mostrar el menú y seleccionar el modo de juego
def mostrar_menu():
    global menu_activo, modo_auto
    pantalla.fill(NEGRO)
    texto = fuente.render("Presiona 'A' para Auto, 'M' para Manual, o 'Q' para Salir", True, BLANCO)
    pantalla.blit(texto, (w // 4, h // 2))
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
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                elif evento.key == pygame.K_q:
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

# Función para reiniciar el juego tras la colisión
def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo
    global bala_vertical, bala_vertical_disparada

    menu_activo = True  # Activar de nuevo el menú
    jugador.x, jugador.y = 50, h - 100  # Reiniciar posición del jugador
    bala.x = w - 50  # Reiniciar posición de la bala
    nave.x, nave.y = w - 100, h - 100  # Reiniciar posición de la nave
    bala_disparada = False
    salto = False
    en_suelo = True

    bala_vertical.x = jugador.x
    bala_vertical.y = 0
    bala_vertical_disparada = False

    # Mostrar los datos recopilados hasta el momento
    print("Datos recopilados para el modelo: ", datos_modelo)

    mostrar_menu()  # Mostrar el menú de nuevo para seleccionar modo


def entrenar_red_neuronal():
    global modelo_nn, scaler_nn, datos_modelo, pausa, modo_auto

    if len(datos_modelo) < 10:
        print("No hay suficientes datos para entrenar la red neuronal.")
        return False

    # Convertir a arrays de NumPy
    datos = np.array(datos_modelo)
    X = datos[:, :2]  # velocidad_bala, distancia
    y = datos[:, 2]   # salto_hecho

    # Normalización
    scaler_nn = StandardScaler()
    X_scaled = scaler_nn.fit_transform(X)

    # Entrenamiento
    modelo_nn = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=500, random_state=42)
    modelo_nn.fit(X_scaled, y)

    print("Red neuronal entrenada con éxito.")
    # Desactivar pausa y activar modo automático
    pausa = False
    modo_auto = True
    return True

def decision_salto_automatico():
    global modelo_nn, scaler_nn, velocidad_bala, jugador, bala
    if not modelo_nn or not scaler_nn:
        print("Modelo no entrenado todavía.")
        return

    distancia = abs(jugador.x - bala.x)
    entrada = np.array([[velocidad_bala, distancia]])
    entrada_norm = scaler_nn.transform(entrada)
    prediccion = modelo_nn.predict(entrada_norm)

    print(f"Predicción: {prediccion[0]} | Vel: {velocidad_bala} | Dist: {distancia}")
    if prediccion[0] == 1:
        realizar_salto()

def realizar_salto():
    global salto, en_suelo
    if en_suelo:
        salto = True
        en_suelo = False


def main():
    global salto, en_suelo, bala_disparada, mover_derecha, mover_izquierda, modo_auto, menu_activo

    reloj = pygame.time.Clock()
    mostrar_menu()  # Mostrar el menú al inicio
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:
                    pausa_juego()
                if evento.key == pygame.K_a and pausa:
                    if entrenar_red_neuronal():
                        menu_activo = False
                if evento.key == pygame.K_q:
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

        # Detectar teclas presionadas en tiempo real (más confiable que KEYDOWN/KEYUP)
        keys = pygame.key.get_pressed()

        if not pausa:
            if modo_auto:
                decision_salto_automatico()
                if salto:
                    manejar_salto()
            else:
                # Movimiento horizontal manual
                if keys[pygame.K_RIGHT]:
                    jugador.x += velocidad
                if keys[pygame.K_LEFT]:
                    jugador.x -= velocidad

                # Mantener al jugador dentro de la pantalla
                jugador.x = max(0, min(jugador.x, w - jugador.width))

                # Salto manual
                if salto:
                    manejar_salto()

                # Guardar datos de entrenamiento solo en modo manual
                guardar_datos()

            # Actualizar disparo de balas si es necesario
            if not bala_disparada:
                disparar_bala()

            # Actualizar el juego (dibujar y colisiones)
            update()

        pygame.display.flip()
        reloj.tick(30)  # Limitar el juego a 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
