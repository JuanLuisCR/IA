from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import re

# Cargar el modelo de embeddings
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Función actualizada para dividir texto en bloques
def dividir_por_bloques(texto_completo):
    # Permite detectar ambos formatos de separador: TEXTO NUEVO o TEXTO <número>
    bloques = re.split(r"(?m)^=== TEXTO(?: NUEVO| \d+) ===\s*", texto_completo)
    bloques = [bloque.strip() for bloque in bloques if bloque.strip()]
    return bloques

# Guardar bloques de texto como referencia
def guardar_textos_divididos(textos, ruta_salida):
    with open(ruta_salida, "w", encoding="utf-8") as f:
        for i, t in enumerate(textos, 1):
            f.write(f"=== TEXTO {i} ===\n{t.strip()}\n\n")

# Embedding + FAISS
def crear_embeddings_y_guardar(ruta_texto, ruta_npy, ruta_faiss):
    with open(ruta_texto, encoding="utf-8") as f:
        contenido = f.read()

    textos = dividir_por_bloques(contenido)

    vectores = model.encode(textos, normalize_embeddings=True)
    np.save(ruta_npy, vectores)

    dim = vectores.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(vectores))
    faiss.write_index(index, ruta_faiss)

    print(f"{len(textos)} bloques procesados.")
    print(f"Embeddings guardados en: {ruta_npy}")
    print(f"Índice FAISS guardado en: {ruta_faiss}")

    ruta_dividido = ruta_texto.replace(".txt", "_dividido.txt")
    guardar_textos_divididos(textos, ruta_dividido)
    print(f"Bloques de texto guardados en: {ruta_dividido}")

# Uso
if __name__ == "__main__":
    base = "C:/Users/juanl/Desktop/Semestre 9/IA/Proyecto 3"

    crear_embeddings_y_guardar(
        os.path.join(base, "data/aborto.txt"),
        os.path.join(base, "embeddings/aborto.npy"),
        os.path.join(base, "embeddings/aborto.faiss")
    )

    crear_embeddings_y_guardar(
        os.path.join(base, "data/eutanasia_dividido.txt"),
        os.path.join(base, "embeddings/eutanasia.npy"),
        os.path.join(base, "embeddings/eutanasia.faiss")
    )