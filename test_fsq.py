from foursquare_service import FoursquareService

service = FoursquareService()

print("--- Iniciando prueba con API 2025 en Pachuca ---")
datos = service.buscar_lugares(20.1287, -98.7303)

if datos and "results" in datos and len(datos["results"]) > 0:
    print(f"✨ ¡ÉXITO! Se encontraron {len(datos['results'])} lugares:")
    for lugar in datos["results"]:
        nombre = lugar.get('name', 'Sin nombre')
        pop = lugar.get('popularity', 'N/A')
        print(f"📍 {nombre} (Popularidad: {pop})")
else:
    print("No se recibieron resultados. Revisa los mensajes de error arriba.")