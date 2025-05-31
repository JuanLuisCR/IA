import pygame

# Inicialización de Pygame
pygame.init()

# Configuración de la ventana
ANCHO_VENTANA = 600
FILAS = 11
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos")

# Definición de colores
COLOR_FONDO = (255, 255, 255)
COLOR_PARED = (50, 50, 50)
COLOR_LINEA = (180, 180, 180)
COLOR_INICIO = (64, 224, 208)     # Turquesa
COLOR_FIN = (255, 20, 147)        # Magenta
COLOR_VISITADO = (96, 130, 182)   # Azul gris
COLOR_CAMINO = (100, 255, 100)      # Amarillo
COLOR_VECINO = (135, 206, 250)    # Celeste
COLOR_ACTUAL = (72, 61, 139)      # Azul oscuro
COLOR_TEXTO = (0, 0, 0)

class Nodo:
    def __init__(self, fila, col, tam):
        self.fila = fila
        self.col = col
        self.x = fila * tam
        self.y = col * tam
        self.ancho = tam
        self.color = COLOR_FONDO
        self.dependiente = None
        self.numero = None
        self.texto = None

    def get_pos(self): return self.fila, self.col
    def es_pared(self): return self.color == COLOR_PARED
    def es_inicio(self): return self.color == COLOR_INICIO
    def es_fin(self): return self.color == COLOR_FIN
    def es_visitado(self): return self.color == COLOR_VISITADO

    def reset(self): self.color = COLOR_FONDO
    def set_inicio(self): self.color = COLOR_INICIO
    def set_fin(self): self.color = COLOR_FIN
    def set_pared(self): self.color = COLOR_PARED
    def set_camino(self): self.color = COLOR_CAMINO
    def set_visitado(self): self.color = COLOR_VISITADO
    def set_vecino(self): self.color = COLOR_VECINO
    def set_actual(self): self.color = COLOR_ACTUAL
    def set_dependiente(self, nodo): self.dependiente = nodo
    def get_dependiente(self): return self.dependiente
    def set_numero(self, n): self.numero = n
    def get_numero(self): return self.numero
    def set_texto(self, t): self.texto = t
    def get_texto(self): return self.texto

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))
        if self.texto:
            fuente = pygame.font.SysFont("Arial", int(self.ancho / 5))
            y = self.y
            for linea in self.texto.split("-"):
                texto = fuente.render(linea, True, COLOR_TEXTO)
                ventana.blit(texto, (self.x + 2, y))
                y += int(self.ancho / 5) + 2

def crear_grid(filas, ancho):
    grid = []
    tam = ancho // filas
    for i in range(filas):
        fila = []
        for j in range(filas):
            nodo = Nodo(i, j, tam)
            fila.append(nodo)
        grid.append(fila)

    num = 1
    for i in range(filas):
        for j in range(filas):
            grid[j][i].set_numero(num)
            num += 1
    return grid

def dibujar_grid(ventana, grid, filas, ancho):
    ventana.fill(COLOR_FONDO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    tam = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, COLOR_LINEA, (0, i * tam), (ancho, i * tam))
        pygame.draw.line(ventana, COLOR_LINEA, (i * tam, 0), (i * tam, ancho))
    pygame.display.update()

def obtener_click(pos, filas, ancho):
    tam = ancho // filas
    y, x = pos
    return y // tam, x // tam

def reconstruir_camino(nodo, grid, ventana):
    while nodo and not nodo.es_inicio():
        nodo.set_camino()
        nodo = nodo.get_dependiente()
        dibujar_grid(ventana, grid, FILAS, ANCHO_VENTANA)

def h(nodo1, nodo2):
    f1, c1 = nodo1.get_pos()
    f2, c2 = nodo2.get_pos()
    return (abs(f1 - f2) + abs(c1 - c2)) * 10

def a_estrella(inicio, fin, grid, ventana):
    abiertos = [inicio]
    g = {inicio: 0}
    f = {inicio: h(inicio, fin)}
    actual = inicio
    anterior = None

    while abiertos:
        abiertos.sort(key=lambda nodo: f.get(nodo, float('inf')))
        actual = abiertos.pop(0)

        if actual == fin:
            reconstruir_camino(actual.get_dependiente(), grid, ventana)
            return

        for d_fila, d_col, costo in [(-1, 0, 10), (1, 0, 10), (0, -1, 10), (0, 1, 10),
                                     (-1, -1, 14), (-1, 1, 14), (1, -1, 14), (1, 1, 14)]:
            nf, nc = actual.fila + d_fila, actual.col + d_col
            if 0 <= nf < FILAS and 0 <= nc < FILAS:
                vecino = grid[nf][nc]
                if not vecino.es_pared():
                    temp_g = g[actual] + costo
                    if temp_g < g.get(vecino, float('inf')):
                        g[vecino] = temp_g
                        f[vecino] = temp_g + h(vecino, fin)
                        vecino.set_dependiente(actual)
                        if vecino not in abiertos:
                            abiertos.append(vecino)
                            vecino.set_texto(f"H={h(vecino, fin)}-G={g[vecino]}-F={f[vecino]}")
                            if not vecino.es_fin():
                                vecino.set_vecino()

        if not actual.es_inicio() and not actual.es_fin():
            actual.set_visitado()
        dibujar_grid(ventana, grid, FILAS, ANCHO_VENTANA)
        pygame.time.delay(70)  # <-- Agrega esta línea aquí


def main(ventana, ancho):
    grid = crear_grid(FILAS, ancho)
    inicio, fin = None, None
    corriendo = True

    while corriendo:
        dibujar_grid(ventana, grid, FILAS, ancho)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            if evento.type == pygame.KEYDOWN and evento.key == pygame.K_SPACE:
                if inicio and fin:
                    a_estrella(inicio, fin, grid, ventana)

            if pygame.mouse.get_pressed()[0]:
                fila, col = obtener_click(pygame.mouse.get_pos(), FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.set_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.set_fin()
                elif nodo != inicio and nodo != fin:
                    nodo.set_pared()

            elif pygame.mouse.get_pressed()[2]:
                fila, col = obtener_click(pygame.mouse.get_pos(), FILAS, ancho)
                nodo = grid[fila][col]
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None
                nodo.reset()

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)
