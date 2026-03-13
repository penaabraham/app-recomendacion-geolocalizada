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
  List filteredRecommendations = [];
  bool isLoading = false;
  String statusMessage = "Busca un producto por nombre";

  final TextEditingController _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> getRecommendations() async {
    final query = _searchController.text.trim();

    setState(() {
      isLoading = true;
      statusMessage = "Obteniendo ubicación...";
    });

    try {
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      setState(() => statusMessage = "Consultando Algoritmo...");

      final url =
          'https://app-recomendacion-geolocalizada.onrender.com/recomendar/1?lat=${position.latitude}&lon=${position.longitude}';

      final response = await http.get(Uri.parse(url));

      if (response.statusCode == 200) {
        final decodedData = json.decode(response.body);

        setState(() {
          if (decodedData is List) {
            recommendations = decodedData;
            _applyFilter(query);
          } else if (decodedData is Map &&
              decodedData.containsKey('error_interno')) {
            statusMessage = "Error Algoritmo: ${decodedData['error_interno']}";
            recommendations = [];
            filteredRecommendations = [];
          } else {
            statusMessage = "Respuesta inesperada del servidor";
          }
          isLoading = false;
        });
      } else {
        setState(() {
          isLoading = false;
          statusMessage = "Error servidor: ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        isLoading = false;
        statusMessage = "Error de conexión: $e";
      });
    }
  }

  void _applyFilter(String query) {
    if (query.isEmpty) {
      filteredRecommendations = List.from(recommendations);
      statusMessage = "Cerca de ti:";
    } else {
      filteredRecommendations = recommendations
          .where((item) => item['name']
              .toString()
              .toLowerCase()
              .contains(query.toLowerCase()))
          .toList();
      statusMessage = filteredRecommendations.isEmpty
          ? "Sin resultados para \"$query\""
          : "Resultados para \"$query\":";
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
          // BARRA DE BÚSQUEDA + BOTÓN GPS
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _searchController,
                    textInputAction: TextInputAction.search,
                    onSubmitted: (_) => getRecommendations(),
                    decoration: InputDecoration(
                      hintText: 'Buscar producto...',
                      prefixIcon:
                          const Icon(Icons.search, color: Colors.indigo),
                      suffixIcon: _searchController.text.isNotEmpty
                          ? IconButton(
                              icon: const Icon(Icons.clear, color: Colors.grey),
                              onPressed: () {
                                _searchController.clear();
                                setState(() => _applyFilter(''));
                              },
                            )
                          : null,
                      filled: true,
                      fillColor: Colors.indigo.withOpacity(0.05),
                      contentPadding: const EdgeInsets.symmetric(vertical: 0),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: BorderSide(
                            color: Colors.indigo.withOpacity(0.3)),
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: BorderSide(
                            color: Colors.indigo.withOpacity(0.3)),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide:
                            const BorderSide(color: Colors.indigo, width: 2),
                      ),
                    ),
                    onChanged: (value) {
                      setState(() {
                        if (recommendations.isNotEmpty) _applyFilter(value);
                      });
                    },
                  ),
                ),
                const SizedBox(width: 10),
                ElevatedButton(
                  onPressed: isLoading ? null : getRecommendations,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.indigo,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: const Icon(Icons.gps_fixed),
                ),
              ],
            ),
          ),

          // MENSAJE DE ESTADO
          Padding(
            padding:
                const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Text(
                statusMessage,
                style:
                    const TextStyle(fontWeight: FontWeight.bold),
              ),
            ),
          ),

          // LISTA DE RESULTADOS
          Expanded(
            child: isLoading
                ? const Center(child: CircularProgressIndicator())
                : filteredRecommendations.isEmpty
                    ? const Center(
                        child: Icon(Icons.search_off,
                            size: 50, color: Colors.grey),
                      )
                    : ListView.builder(
                        itemCount: filteredRecommendations.length,
                        itemBuilder: (context, index) {
                          final item = filteredRecommendations[index];
                          final double rating =
                              double.tryParse(
                                      item['avg_rating'].toString()) ??
                                  0.0;
                          return Card(
                            elevation: 3,
                            margin: const EdgeInsets.symmetric(
                                horizontal: 15, vertical: 8),
                            shape: RoundedRectangleBorder(
                                borderRadius:
                                    BorderRadius.circular(12)),
                            child: ListTile(
                              leading: CircleAvatar(
                                backgroundColor:
                                    Colors.indigo.withOpacity(0.1),
                                child: const Icon(Icons.shopping_bag,
                                    color: Colors.indigo),
                              ),
                              title: Text(item['name'],
                                  style: const TextStyle(
                                      fontWeight: FontWeight.bold)),
                              subtitle: Column(
                                crossAxisAlignment:
                                    CrossAxisAlignment.start,
                                children: [
                                  const SizedBox(height: 4),
                                  Row(
                                    children: [
                                      const Icon(Icons.location_on,
                                          size: 14, color: Colors.red),
                                      Text(" ${item['distance']}"),
                                      const SizedBox(width: 15),
                                      const Icon(Icons.star,
                                          size: 14,
                                          color: Colors.amber),
                                      Text(
                                          " ${rating.toStringAsFixed(1)}"),
                                    ],
                                  ),
                                  const SizedBox(height: 8),
                                  Wrap(
                                    spacing: 5,
                                    children: (item['reason'] as String)
                                        .split(',')
                                        .map<String>(
                                            (r) => r.trim())
                                        .where((r) => r.isNotEmpty)
                                        .map<Widget>((r) {
                                      return Container(
                                        padding:
                                            const EdgeInsets.symmetric(
                                                horizontal: 8,
                                                vertical: 2),
                                        decoration: BoxDecoration(
                                          color: Colors.green
                                              .withOpacity(0.1),
                                          borderRadius:
                                              BorderRadius.circular(
                                                  10),
                                          border: Border.all(
                                              color: Colors.green
                                                  .withOpacity(0.3)),
                                        ),
                                        child: Text(
                                          "✓ $r",
                                          style: const TextStyle(
                                              fontSize: 10,
                                              color: Colors.green),
                                        ),
                                      );
                                    }).toList(),
                                  ),
                                ],
                              ),
                            ),
                          );
                        },
                      ),
          ),
        ],
      ),
    );
  }
}