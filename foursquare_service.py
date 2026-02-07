import requests
import os
from dotenv import load_dotenv

load_dotenv()

class FoursquareService:
    def __init__(self):
        self.service_key = os.getenv("FOURSQUARE_API_KEY")
        # CAMBIO 1: Nueva URL base sin el segmento /v3/
        self.base_url = "https://places-api.foursquare.com/places/search"
        
        # CAMBIO 2: Formato Bearer y versión con guiones
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.service_key}",
            "X-Places-Api-Version": "2025-06-17" 
        }

    def buscar_lugares(self, lat: float, lon: float, radio: int = 2000):
        # Reducimos los campos al mínimo absoluto (Nombre y ID) 
        # para intentar que no nos cobren créditos Pro/Premium
        params = {
            "ll": f"{lat},{lon}",
            "radius": str(radio),
            "limit": "3",
            "fields": "name,location" # Quitamos popularity y rating por ahora
        }