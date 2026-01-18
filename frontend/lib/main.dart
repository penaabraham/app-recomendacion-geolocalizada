import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Recomendador Híbrido',
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
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

  // FUNCIÓN CLAVE: Llama a tu servidor Python
  Future<void> fetchRecommendations() async {
    setState(() => isLoading = true);
    try {
      // USAMOS TU IP 192.168.1.105
      final response = await http.get(
        Uri.parse('http://192.168.1.105:8000/recomendar/1')
      );

      if (response.statusCode == 200) {
        setState(() {
          recommendations = json.decode(response.body);
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() => isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: No se pudo conectar al servidor')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Mis Recomendaciones')),
      body: isLoading 
        ? const Center(child: CircularProgressIndicator())
        : ListView.builder(
            itemCount: recommendations.length,
            itemBuilder: (context, index) {
              final item = recommendations[index];
              return ListTile(
                leading: const Icon(Icons.shopping_cart, color: Colors.blue),
                title: Text(item['name'] ?? 'Producto'),
                subtitle: Text('Distancia: ${item['distance']}'),
                trailing: const Icon(Icons.arrow_forward_ios, size: 16),
              );
            },
          ),
      floatingActionButton: FloatingActionButton(
        onPressed: fetchRecommendations,
        child: const Icon(Icons.refresh),
      ),
    );
  }
}