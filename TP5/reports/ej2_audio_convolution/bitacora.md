# Bitácora: comparación convolucional de audio

15/11/2026. El usuario autorizó implementar la comparación propuesta de VAE
denso frente a convolucional, manteniendo códec y pérdida.

- Cuatro configuraciones principales, tres semillas cada una, 200 épocas.
- Se reutiliza la arquitectura espacial existente con mapas 128×64×1.
  Hay convoluciones en encoder y decoder; no se cambiaron fórmulas del modelo.
- Validación: notas 51/57/63, variantes v00/v20/v39; exclusión de 0,75 semitonos.
  Las finales 55/61/67 continúan excluidas y sin evaluar.
- Auditoría: datos idénticos por condición para ambos modelos, con hashes en
  `protocol.json`; 1.420/1.120 muestras. Parámetros: 8.535.328/8.537.809.
- Las cuatro pruebas integrales de dos épocas terminaron. Están separadas en
  carpetas `_smoke`; no son evidencia de desempeño musical.
- Batería general: **237 pruebas pasaron en 366,70 s**. La prueba adicional del
  lector de resultados y la validación final del protocolo se ejecutaron después:
  **5 pruebas nuevas pasaron en 10,34 s**. Subconjunto de arquitectura/persistencia/
  checkpoints/análisis: 46 pruebas pasaron en 19,84 s (se superponen con la batería).
- Se inició `bash scripts/run_audio_convolution.sh 3` en segundo plano. Tres
  trabajadores activos; la fase inicial es el control completo. Hay checkpoints
  persistidos. La salida está en `results/audio_convolution/run.log`.
- El informe se genera tras cada fase. Al momento de esta entrada todavía no
  terminaron corridas principales: **no hay conclusión musical**.
- Se verificó que `losses.py`, `autoencoder.py`, `variational.py` y la presentación
  vigente no presentan cambios. Este trabajo no abre otra búsqueda de figuras.

Para continuar tras interrumpir: repetir el ejecutor sin `--force`. Un lock evita
duplicar ejecuciones. Se conserva el estado de Adam, RNG y época del checkpoint.
Detalles y comandos: `docs/audio_convolution.md`.
