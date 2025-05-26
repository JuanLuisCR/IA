import pygame
import random
import pandas as pd
from joblib import load
from sklearn.tree import DecisionTreeClassifier

pygame.init()

w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo Horizontal y Caída Vertical")

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

modelo = load("C:/Users/juanl/Desktop/Semestre 9/IA/pygamesc/modelo_arbol_decision.pkl")

def disparar_bala_h():
    global bala_h_disparada, velocidad_bala_h
    if not bala_h_disparada:
        velocidad_bala_h = random.randint(-8, -3)
        bala_h_disparada = True

def disparar_bala_v():
    global bala_v_disparada, velocidad_bala_v
    if not bala_v_disparada:
        velocidad_bala_v = random.randint(1, 3)  # Velocidad positiva para caída
        bala_v_disparada = True

def reset_bala_h():
    global bala_h, bala_h_disparada
    bala_h.x = w - 50
    bala_h_disparada = False

def reset_bala_v():
    global bala_v, bala_v_disparada
    bala_v.y = 0  # Desde arriba
    bala_v.x = random.randint(0, w - 16)  # Posición horizontal aleatoria
    bala_v_disparada = False


    

def guardar_csv():
    df = pd.DataFrame(datos_modelo, columns=[
        "velocidad", "jugador_x", "bala_h_x", "bala_v_x", "salto", "accion_horizontal"
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
    salto_hecho = 1 if salto else 0

    if pygame.key.get_pressed()[pygame.K_LEFT]:
        accion_horizontal = -1
    elif pygame.key.get_pressed()[pygame.K_RIGHT]:
        accion_horizontal = 1
    else:
        accion_horizontal = 0

    datos_modelo.append((
        velocidad_bala_h,
        jugador.x,
        bala_h.x,
        bala_v.x,
        accion_horizontal,
        salto_hecho
    ))

def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado. Datos:", datos_modelo)
    else:
        print("Reanudado.")

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
                    print("Juego terminado. Datos:", datos_modelo)
                    pygame.quit()
                    guardar_csv()
                    exit()

def reiniciar_juego():
    global menu_activo, jugador, bala_h, bala_v, nave, bala_h_disparada, bala_v_disparada, salto, en_suelo
    menu_activo = True
    jugador.x, jugador.y = 50, h - 100
    bala_h.x = w - 50
    bala_v.x = random.randint(0, w - 16)
    bala_v.y = 0
    nave.x, nave.y = w - 100, h - 100
    bala_h_disparada = False
    bala_v_disparada = False
    salto = False
    en_suelo = True
    print("Datos recopilados:", datos_modelo)
    mostrar_menu()

def update():
    global frame_count, current_frame, fondo_x1, fondo_x2, bala_v

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
        print("Colisión detectada!")
        guardar_csv()
        reiniciar_juego()

def main():
    global salto, en_suelo, bala_h_disparada, bala_v_disparada

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
                    print("Juego terminado. Datos:", datos_modelo)
                    pygame.quit()
                    guardar_csv()
                    exit()

        if not pausa:
            teclas = pygame.key.get_pressed()
            if not modo_auto:
                if teclas[pygame.K_LEFT]:
                    jugador.x -= 5
                if teclas[pygame.K_RIGHT]:
                    jugador.x += 5

            if not bala_v_disparada:
                disparar_bala_v()
            if not bala_h_disparada:
                disparar_bala_h()

            if modo_auto:
                entrada = [[
                    velocidad_bala_h,
                    jugador.x,
                    bala_h.x,
                    bala_v.x
                ]]

                accion_h, accion_salto = modelo.predict(entrada)[0]

                if accion_h == -1:
                    jugador.x -= 5
                elif accion_h == 1:
                    jugador.x += 5

                if accion_salto == 1 and en_suelo:
                    salto = True
                    en_suelo = False

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