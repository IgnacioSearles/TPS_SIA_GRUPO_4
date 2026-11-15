# Propuesta adicional, 15/11/2026

Abrir `con_agregados.html`; `index.html` conserva exactamente la presentación vigente.
55 páginas originales sin cambios, en el mismo orden relativo; siete agregados (62 total).

## Páginas nuevas en la propuesta

- 37: El muestreo también cambia el resultado
- 46: El mismo mecanismo variacional con figuras RGB
- 47: Qué activan los filtros de la CNN
- 48: Limpieza y geometría son propiedades diferentes
- 60: Dónde elegimos los códigos
- 61: Controles y alcance de los resultados
- 62: La segunda capa: los 32 mapas completos

Las cuatro primeras se integran a la explicación; las tres últimas son respaldo.
Todos los agregados incluyen notas del orador. No se reemplazan las notas anteriores.
Los botones de audio conservan el controlador existente.

## Fuentes e interpretación

- `reports/ej2_cierre_generativo_2026-11-15/guion_solo_agregados.md`: motivación y guion detallado.
- `reports/ej2_activaciones_cnn/`: inferencia real sin entrenar; mapas completos y pesos verificados.
- `results/nsynth_sampling_control/`: control complementario nuevo, una semilla, tres distribuciones.
- `results/art_latent_mixture/`: GMM sobre códigos de entrenamiento, sin etiquetas; tres semillas.
- `reports/ej2_cierre_generativo_2026-11-15/`: tablas, figuras y alcance del cierre.

El AE con densidad ajustada conserva su entrenamiento determinista. No es un VAE.
Las GMM cambian el muestreo posterior al entrenamiento; no mejoran retroactivamente el prior N(0,I).
Limpieza no equivale a geometría correcta; cercanía al entrenamiento no demuestra novedad.
RGB y audio comparten mecanismo variacional, sin compartir pesos ni comparar sus MSE.
La prueba final quedó sin abrir. No se hicieron nuevos entrenamientos.

## Reproducir y verificar

Desde TP5: `PYTHONPATH=. .venv/bin/python -m scripts.export_cnn_activations`
y `PYTHONPATH=. .venv/bin/python -m scripts.build_presentation_additions`.
El segundo comando escribe dos paquetes en `results/slides_review_2026-11-15/`
y `verificacion.json` con hashes de todos los archivos originales.
Verifica 55 secciones literales preservadas y 62 finales, referencias locales existentes,
y que quitar las siete inserciones y el vínculo CSS reconstruye el HTML original exactamente.
