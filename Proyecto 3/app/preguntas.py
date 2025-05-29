from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import re

# Función para dividir texto en bloques igual que al crear embeddings
def dividir_por_bloques(texto_completo):
    bloques = re.split(r"(?m)^=== TEXTO(?: NUEVO| \d+) ===\s*", texto_completo)
    bloques = [b.strip() for b in bloques if b.strip()]
    return bloques

def limpiar_pregunta(p):
    return re.sub(r"\W+", " ", p.strip().lower())

# Cargar embeddings del aborto
vectores_aborto = np.load("C:/Users/juanl/Desktop/Semestre 9/IA/Proyecto 3/embeddings/aborto.npy")
with open("C:/Users/juanl/Desktop/Semestre 9/IA/Proyecto 3/data/aborto.txt", encoding="utf-8") as f:
    textos_aborto = dividir_por_bloques(f.read())
index_aborto = faiss.read_index("C:/Users/juanl/Desktop/Semestre 9/IA/Proyecto 3/embeddings/aborto.faiss")

# Cargar embeddings de la eutanasia
vectores_eutanasia = np.load("C:/Users/juanl/Desktop/Semestre 9/IA/Proyecto 3/embeddings/eutanasia.npy")
with open("C:/Users/juanl/Desktop/Semestre 9/IA/Proyecto 3/data/eutanasia.txt", encoding="utf-8") as f:
    textos_eutanasia = dividir_por_bloques(f.read())
index_eutanasia = faiss.read_index("C:/Users/juanl/Desktop/Semestre 9/IA/Proyecto 3/embeddings/eutanasia.faiss")

# Modelo embeddings para pregunta
modelo_embed = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

cache_respuestas = {}

# Calcular embedding promedio de cada tema (una sola vez)
promedio_aborto = np.mean(vectores_aborto, axis=0)
promedio_eutanasia = np.mean(vectores_eutanasia, axis=0)

def distancia_coseno(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return 1 - np.dot(a_norm, b_norm)

def detectar_tema(embedding_pregunta):
    dist_aborto = distancia_coseno(embedding_pregunta, promedio_aborto)
    dist_eutanasia = distancia_coseno(embedding_pregunta, promedio_eutanasia)
    # print(f"Distancias — Aborto: {dist_aborto:.4f} | Eutanasia: {dist_eutanasia:.4f}")
    if dist_aborto < dist_eutanasia:
        return "aborto"
    else:
        return "eutanasia"

def responder_pregunta(pregunta, k=3):
    pregunta_clave = limpiar_pregunta(pregunta)
    if pregunta_clave in cache_respuestas:
        print("[Respuesta desde caché]")
        return cache_respuestas[pregunta_clave]

    # Embedding normalizado para coherencia con FAISS
    embedding_pregunta = modelo_embed.encode([pregunta], normalize_embeddings=True)[0]

    tema = detectar_tema(embedding_pregunta)
    print(f"[Tema detectado: {tema}]")

    if tema == "aborto":
        D, I = index_aborto.search(embedding_pregunta[None, :], k)
        textos_recuperados = [textos_aborto[i].strip() for i in I[0]]
    else:
        D, I = index_eutanasia.search(embedding_pregunta[None, :], k)
        textos_recuperados = [textos_eutanasia[i].strip() for i in I[0]]

    contexto = "\n".join(textos_recuperados)

    if len(contexto.strip()) < 20:
        return "Lo siento, no tengo información suficiente para responder a esa pregunta."

    # print("\n[Contexto usado para responder]:")
    # for i, texto in enumerate(textos_recuperados, 1):
    #     print(f"{i}. {texto}")

    prompt = f"""Contexto relevante extraído exclusivamente de mis textos:
{contexto}

Pregunta:
{pregunta}

RESPONDE SOLO con la información proporcionada en el contexto anterior.
Sé breve y preciso."""

    try:
        respuesta = ollama.chat(model='llama3:8b', messages=[
            {'role': 'system', 'content': 'Eres un asistente experto que responde solo con la información dada en el contexto.'},
            {'role': 'user', 'content': prompt}
        ])
        contenido = respuesta['message']['content']
        cache_respuestas[pregunta_clave] = contenido
        return contenido
    except Exception as e:
        return f"Error al generar respuesta: {e}"

def main():
    print("Chatbot experto en ética y tecnología. Escribe 'salir' para terminar.")
    while True:
        pregunta = input("\nPregunta: ").strip()
        if pregunta.lower() == 'salir':
            print("Adiós.")
            break
        if not pregunta:
            print("Por favor ingresa una pregunta válida.")
            continue
        respuesta = responder_pregunta(pregunta)
        print("\nRespuesta:\n", respuesta)

if __name__ == "__main__":
    main()
