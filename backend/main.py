from fastapi import FastAPI
import ALGORITMO_PROPUESTO as algoritmo # Importa tu script

app = FastAPI()

@app.get("/recomendar/{user_id}")
def obtener_recomendacion(user_id: int):
    # Llama a la función que ya tienes en tu script
    resultado = algoritmo.recommend(user_id)
    # Convertir el DataFrame de pandas a un diccionario para enviarlo
    return resultado.to_dict(orient='records')