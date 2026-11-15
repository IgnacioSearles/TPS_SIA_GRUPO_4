# Propuesta con arquitecturas exactas

Abrir `con_agregados_arquitecturas.html` desde esta carpeta. Usa los mismos assets y reproductor de audio de la presentación, sin reemplazar `index.html`.

La propuesta tiene 73 diapositivas. Conserva literalmente las 69 de `con_agregados_hilo_completo.html` y agrega:

- 53: definición de convolución, kernel, padding, stride y dimensiones de salida.
- 54: encoder denso inicial, primera CNN y CNN ancha/refinada de figuras.
- 55: decoders exactos y capas adicionales de refinamiento.
- 73: encoder y decoder convolucionales de audio y su control denso.

Las notas incluyen fuentes de configuraciones, activaciones y límites de cada comparación. La operación implementada es correlación cruzada 3×3. No usamos max pooling ni convolución transpuesta. Upsample repite píxeles, mientras que los filtros siguientes aprenden pesos.

Regeneración: `python scripts/add_formal_architectures.py`, desde TP5. La verificación de preservación queda en `reports/figure_architectures/verification.json`.

Audio, corte de revisión: terminaron 3/12 corridas, las CNN del control con notas presentes. Reconstrucción directa: altura y criterio espectral 27/27. Interpolación: altura 15/27, criterio espectral 11/27 y ambos 11/27. La mezcla de espectrogramas obtiene altura 27/27 pero criterio espectral 0/27. No hay todavía comparación completa con el denso ni evaluación CNN con notas omitidas. No se declara mejora. Informe: `reports/ej2_audio_convolution/informe.md`.
