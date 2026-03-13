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
  // Lista completa descargada del servidor
  List _allRecommendations = [];

  // Slice visible en pantalla (crece de 10 en 10)
  List _visibleRecommendations = [];

  static const int _pageSize = 10;

  bool _isLoading = false;         // carga inicial
  bool _isLoadingMore = false;     // carga paginada
  String _statusMessage = "Busca un producto por nombre";

  final TextEditingController _searchController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onScroll);
  }

  @override
  void dispose() {
    _searchController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // Detecta cuando el usuario llega al final de la lista
  void _onScroll() {
    if (_scrollController.position.pixels >=
        _scrollController.position.maxScrollExtent - 200) {
      _loadMore();
    }
  }

  // Agrega los siguientes _pageSize elementos a la lista visible
  void _loadMore() {
    if (_isLoadingMore) return;
    if (_visibleRecommendations.length >= _allRecommendations.length) return;

    setState(() => _isLoadingMore = true);

    final nextEnd = (_visibleRecommendations.length + _pageSize)
        .clamp(0, _allRecommendations.length);

    Future.delayed(const Duration(milliseconds: 300), () {
      if (!mounted) return;
      setState(() {
        _visibleRecommendations =
            _allRecommendations.sublist(0, nextEnd);
        _isLoadingMore = false;
      });
    });
  }

  // Aplica el filtro de texto y reinicia la paginación al primer bloque
  void _applyFilter(String query) {
    final base = query.isEmpty
        ? List.from(_allRecommendations)
        : _allRecommendations
            .where((item) => item['name']
                .toString()
                .toLowerCase()
                .contains(query.toLowerCase()))
            .toList();

    // Guardamos la lista filtrada completa en _allRecommendations
    // solo cuando hay búsqueda activa; de lo contrario se usa la original.
    // Para simplificar, reasignamos _allRecommendations al filtrado:
    _allRecommendations    = base;
    _visibleRecommendations = base.take(_pageSize).toList();

    if (query.isEmpty) {
      _statusMessage = "Cerca de ti:";
    } else {
      _statusMessage = base.isEmpty
          ? "Sin resultados para \"$query\""
          : "Resultados para \"$query\" (${base.length}):";
    }
  }

  Future<void> _getRecommendations() async {
    final query = _searchController.text.trim();

    setState(() {
      _isLoading     = true;
      _statusMessage = "Obteniendo ubicación...";
    });

    try {
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      setState(() => _statusMessage = "Consultando Algoritmo...");

      // Sin parámetro limit → el backend devuelve todos los productos
      final url =
          'https://app-recomendacion-geolocalizada.onrender.com/recomendar/1'
          '?lat=${position.latitude}&lon=${position.longitude}';

      final response = await http.get(Uri.parse(url));

      if (response.statusCode == 200) {
        final decodedData = json.decode(response.body);

        setState(() {
          if (decodedData is List) {
            // Guardamos todo y aplicamos filtro + primera página
            _allRecommendations = decodedData;
            _applyFilter(query);
          } else if (decodedData is Map &&
              decodedData.containsKey('error_interno')) {
            _statusMessage          = "Error: ${decodedData['error_interno']}";
            _allRecommendations     = [];
            _visibleRecommendations = [];
          } else {
            _statusMessage = "Respuesta inesperada del servidor";
          }
          _isLoading = false;
        });
      } else {
        setState(() {
          _isLoading     = false;
          _statusMessage = "Error servidor: ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        _isLoading     = false;
        _statusMessage = "Error de conexión: $e";
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
          // BARRA DE BÚSQUEDA + BOTÓN GPS
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _searchController,
                    textInputAction: TextInputAction.search,
                    onSubmitted: (_) => _getRecommendations(),
                    decoration: InputDecoration(
                      hintText: 'Buscar producto...',
                      prefixIcon:
                          const Icon(Icons.search, color: Colors.indigo),
                      suffixIcon: _searchController.text.isNotEmpty
                          ? IconButton(
                              icon: const Icon(Icons.clear,
                                  color: Colors.grey),
                              onPressed: () {
                                _searchController.clear();
                                // Si ya hay datos, re-aplicar sin filtro
                                if (_allRecommendations.isNotEmpty) {
                                  setState(() => _applyFilter(''));
                                }
                              },
                            )
                          : null,
                      filled: true,
                      fillColor: Colors.indigo.withOpacity(0.05),
                      contentPadding:
                          const EdgeInsets.symmetric(vertical: 0),
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
                        borderSide: const BorderSide(
                            color: Colors.indigo, width: 2),
                      ),
                    ),
                    onChanged: (value) {
                      // Filtro en tiempo real sobre los datos ya descargados
                      if (_allRecommendations.isNotEmpty) {
                        setState(() => _applyFilter(value));
                      }
                    },
                  ),
                ),
                const SizedBox(width: 10),
                ElevatedButton(
                  onPressed: _isLoading ? null : _getRecommendations,
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
                _statusMessage,
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
            ),
          ),

          // LISTA DE RESULTADOS
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : _visibleRecommendations.isEmpty
                    ? const Center(
                        child: Icon(Icons.search_off,
                            size: 50, color: Colors.grey),
                      )
                    : ListView.builder(
                        controller: _scrollController,
                        // +1 para el indicador de carga al final
                        itemCount: _visibleRecommendations.length + 1,
                        itemBuilder: (context, index) {
                          // Último elemento: spinner o mensaje "ya no hay más"
                          if (index == _visibleRecommendations.length) {
                            if (_isLoadingMore) {
                              return const Padding(
                                padding: EdgeInsets.all(16),
                                child: Center(
                                    child: CircularProgressIndicator()),
                              );
                            }
                            if (_visibleRecommendations.length >=
                                _allRecommendations.length) {
                              return Padding(
                                padding: const EdgeInsets.all(16),
                                child: Center(
                                  child: Text(
                                    "— ${_allRecommendations.length} resultados en total —",
                                    style: const TextStyle(
                                        color: Colors.grey, fontSize: 12),
                                  ),
                                ),
                              );
                            }
                            return const SizedBox.shrink();
                          }

                          // Tarjeta normal
                          final item = _visibleRecommendations[index];
                          final double rating =
                              double.tryParse(
                                      item['avg_rating'].toString()) ??
                                  0.0;

                          return Card(
                            elevation: 3,
                            margin: const EdgeInsets.symmetric(
                                horizontal: 15, vertical: 8),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
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
                                        .map<String>((r) => r.trim())
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
                                              BorderRadius.circular(10),
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