from social_service import MastodonService

service = MastodonService()
hashtag = "technology"
datos = service.obtener_propuesta_datos(hashtag)

print(f"=== REPORTE DE DATOS: #{hashtag} ===")
for p in datos:
    print(f"\n--- Post ID: {p['id_post']} ---")
    print(f"🕒 Publicado el: {p['fecha']}")
    print(f"💬 Contenido: {p['texto_limpio'][:100]}...")
    print(f"📈 Relevancia (Interacciones): {p['popularidad']}")
    print(f"👤 Perfil Usuario: {p['usuario']['nombre']}")
    print(f"📖 Bio para Vectorización: {p['usuario']['bio'][:80]}...")
    print(f"🏷️ Keywords: {p['etiquetas']}")
    print("-" * 40)