from social_service import MastodonService

service = MastodonService()

# Buscaremos algo global primero para asegurar que funciona, luego probamos con #Pachuca
hashtag = "Hidalgo" 
posts = service.obtener_posts_tiempo_real(hashtag)

print(f"--- Resultados para #{hashtag} ---")
if posts:
    for i, post in enumerate(posts):
        # Limpiamos un poco el contenido (viene en HTML)
        contenido = post['content'][:100].replace('<p>', '').replace('</p>', '')
        autor = post['account']['display_name']
        print(f"{i+1}. 👤 {autor}: {contenido}...")
else:
    print("No se encontraron posts recientes.")