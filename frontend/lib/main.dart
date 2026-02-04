import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:geolocator/geolocator.dart';
import 'dart:convert';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Recomendador GPS',
      theme: ThemeData(primarySwatch: Colors.indigo, useMaterial3: true),
      home: const RecsPage(),
    );
  }
}

class RecsPage extends StatefulWidget {
  const RecsPage({super.key});
  @override
  State<RecsPage> createState() => _RecsPageState();
}

class _RecsPageState extends State<RecsPage> {
  List recommendations = [];
  bool isLoading = false;
  String statusMessage = "Presiona el botón para buscar";

  // FUNCIÓN PARA OBTENER EL GPS Y LLAMAR AL BACKEND
  Future<void> getRecommendations() async {
    setState(() {
      isLoading = true;
      statusMessage = "Obteniendo ubicación...";
    });

    try {
      // 1. Verificar/Pedir permisos de GPS
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      // 2. Obtener coordenadas actuales
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high
      );

      setState(() => statusMessage = "Consultando Algoritmo...");

      // Enviamos lat y lon como parámetros en la URL
      final url = 'https://app-recomendacion-geolocalizada.onrender.com/recomendar/1?lat=${position.latitude}&lon=${position.longitude}';
      
      final response = await http.get(Uri.parse(url));

      if (response.statusCode == 200) {
        setState(() {
          recommendations = json.decode(response.body);
          isLoading = false;
          statusMessage = "Cerca de ti:";
        });
      }
    } catch (e) {
      setState(() {
        isLoading = false;
        statusMessage = "Error: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Recomendaciones Reales'),
        backgroundColor: Colors.indigo,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(statusMessage, style: const TextStyle(fontWeight: FontWeight.bold)),
          ),
          Expanded(
            child: isLoading
                ? const Center(child: CircularProgressIndicator())
                : recommendations.isEmpty
                    ? const Center(child: Icon(Icons.location_off, size: 50, color: Colors.grey))
                    : ListView.builder(
                        itemCount: recommendations.length,
                        itemBuilder: (context, index) {
                          final item = recommendations[index];
                          return Card(
                            margin: const EdgeInsets.symmetric(horizontal: 15, vertical: 5),
                            child: ListTile(
                              leading: const Icon(Icons.place, color: Colors.redAccent),
                              title: Text(item['name'], style: const TextStyle(fontWeight: FontWeight.bold)),
                              subtitle: Text("Distancia calculada: ${item['distance']}"),
                            ),
                          );
                        },
                      ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: getRecommendations,
        label: const Text("Actualizar con GPS"),
        icon: const Icon(Icons.gps_fixed),
      ),
    );
  }
}