# Hilo conductor reforzado

La revisión conserva las 58 diapositivas del remoto y las tres aclaraciones de audio.
Solo reemplaza el bloque de figuras: las páginas 50–54 de la versión de 66 páginas
se convierten en la secuencia 50–57 de la versión de **69 páginas**. Las otras
61 secciones quedan literalmente iguales, en el mismo orden, con las mismas notas.

## Lógica tomada de la presentación

El ejercicio 1 primero muestra resultados y problemas de las redes, después
prueba cambios y recién entonces presenta la arquitectura elegida. El bloque
visual anterior no seguía esa lógica: hablaba de agregar «otra convolución» sin
haber presentado el VAE denso, su resultado ni la primera arquitectura CNN.

La secuencia nueva distingue observación, hipótesis, exploración y control:

1. **50 — Pregunta y datos.** Partimos de un VAE denso del mismo tipo que el de
   audio, entrenado independientemente con figuras. Buscamos formas reconocibles
   al generar sin entrada. Se anticipa que pasamos de 1.024 a 4.096 imágenes.
2. **51 — Resultado original.** Se muestran las reconstrucciones y las primeras
   generaciones del VAE denso anterior a las CNN. El ruido de fondo y los contornos
   difusos motivan una hipótesis de localidad. No se inventa una imagen inicial.
3. **52 — Qué aporta una convolución.** Se explica una ventana 3×3 y el uso de
   los mismos pesos en distintas posiciones. Se ubican las convoluciones en
   encoder y decoder, manteniendo el esquema variacional y la fórmula del error.
4. **53 — Primeras CNN.** Las galerías de los pilotos básico y ancho muestran
   que aún había bordes difusos. Se explica que luego exploramos entrenamiento,
   peso de KL y refinamiento; esos cambios no se confunden con una ablación.
5. **54 — Control aislado.** Recién aquí se introduce una segunda capa de
   refinamiento sobre una CNN ya afinada. Se conserva el gráfico pareado de tres
   semillas y el resultado 175/192 a 182/192 imágenes limpias.
6. **55 — Límite observado.** Las imágenes muestran que mejorar limpieza no
   exige esquinas, lados rectos o curvas regulares. No hay tasa geométrica inventada.
7. **56 — Otra hipótesis, otro experimento.** Se cambia el sorteo de códigos con
   GMM, manteniendo decoders. Se distinguen esos resultados del control de capas.
8. **57 — Regreso a audio.** La conclusión separa mejoras parciales de limpieza
   de la estructura completa que queríamos conservar. No se presenta como éxito
   de notas omitidas ni como demostración de que un VAE jamás pueda aprender formas.

## Evidencia del punto de partida

Modelo: `results/ej2_width_art_vae_wide/base__seed=7`, entrenado antes de las CNN:
capas densas [256,128], latente 8, 1.024 imágenes, 200 épocas y beta 0,5.
Se reconstruyen referencias del protocolo `art_width_data/validation` sin entrenar
ni cambiar los pesos; su hash se verifica antes y después. Las generaciones son
los primeros cuatro índices guardados, de muestreo estándar, sin selección por calidad.
La tercera fila de la imagen no corresponde a los originales de arriba.

Los primeros pilotos CNN son `cnn4k_100` y `cnn4k_wide100`, con sus cuatro primeros
índices. Entre el denso histórico y estos pilotos también cambian el dataset y
el presupuesto. Por ello se presentan como etapas exploratorias, **no como una
medición causal de la ventaja de CNN**. El control posterior sí comparte pesos
padres, datos, pasos e hiperparámetros y cambia únicamente una capa.

No se ejecutaron nuevos entrenamientos. `verification.json` conserva configuraciones,
hashes, rutas e índices, además de la prueba de preservación del resto del deck.
La entrega HTML incorpora los mismos 80 botones de audio, sin ZIP ni dependencias.
