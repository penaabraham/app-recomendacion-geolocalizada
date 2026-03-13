from fastapi import FastAPI, Query
import ALGORITMO_PROPUESTO as algoritmo 

app = FastAPI()

# 1. Unificamos todo en una sola ruta (endpoint)
@app.get("/recomendar/{user_id}")
def obtener_recomendacion(
    user_id: int, 
    lat: float = Query(None), 
    lon: float = Query(None),
    limit: int = Query(None),  # None = todos los productos
):
    try:
        # 2. Definimos coordenadas: si vienen del celular se usan, si no, valores fijos
        mi_latitud = lat if lat is not None else 19.4326  
        mi_longitud = lon if lon is not None else -99.1332

        # 3. Llamamos a la función de tu algoritmo (3 parámetros)
        resultado = algoritmo.recommend(
            user_id=user_id, 
            lat_manual=mi_latitud, 
            lon_manual=mi_longitud,
            top_n=limit,
            )
        
        # 4. Convertimos a diccionario (records) para que Flutter lo entienda
        return resultado.to_dict(orient='records')
    
    except Exception as e:
        return {"error_interno": str(e), "tipo": str(type(e))}

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "API de Recomendación Geolocalizada funcionando correctamente",
        "version": "1.0.0"
    }