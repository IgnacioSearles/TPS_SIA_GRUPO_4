# Propuesta vigente: solamente agregar

Instrucción del usuario: no borrar ni modificar la presentación existente.
La comparación con `origin/main` confirmó que no había diapositivas, notas,
estilos o assets editados por el asistente que revertir. Los cambios anteriores
en la presentación habían entrado desde el remoto solicitado.

La propuesta anterior de actualizar o reemplazar la diapositiva 36 **no se
ejecutó y queda descartada**. Este documento la sustituye como plan vigente,
sin borrar el registro histórico. La regla también queda en `presentation/AGENTS.md`.

## Agregados propuestos, sin tocar las páginas actuales

1. **Control complementario de muestreo.** Una diapositiva nueva después de la
   comparación AE/VAE, con los tres métodos de la reproducción nueva de NSynth.
   Debe mostrar sus propios ejemplos completos y metadatos; no se pega una fila
   de otra corrida sobre los assets anteriores. Explica que ajustar una densidad
   al AE no equivale a entrenar un VAE y que muestrear desde N(0,I) no es un
   control suficiente para descartar toda generación con AE.
2. **Figuras RGB como otra aplicación.** Diapositiva nueva con originales,
   reconstrucciones y generaciones, usando los índices prefijados y explicitando
   que los pesos, arquitectura y beta son propios de ese modelo. La relación con
   audio es el mecanismo variacional y la pregunta generativa, no comparar sus MSE.
3. **Alcance de las métricas y del prior.** Diapositiva nueva que separa limpieza,
   geometría y cercanía al entrenamiento. Puede incorporar el diagnóstico nuevo:
   mismo decoder con prior estándar, gaussiana de momentos y mezcla posterior.
   La reconstrucción desde medias es una referencia, no una generación nueva.
   El objetivo es explicar dónde aparecen los errores, no reemplazar el generador
   estándar por uno ajustado sin declarar el cambio.
4. **Respaldo adicional.** Nuevas páginas de apéndice con tablas por semilla,
   todos los criterios y límites. Si una afirmación existente necesita matiz,
   agregar una aclaración en una página nueva; no reescribir sus textos o notas.

Todo asset nuevo tendrá una ruta nueva, sin sobreescribir imágenes o sonidos
actuales. Las secciones existentes conservarán su contenido y su orden relativo.
Todavía no se agregaron diapositivas: este es el plan aditivo para incorporarlas.
