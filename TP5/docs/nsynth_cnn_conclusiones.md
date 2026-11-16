# Conclusiones de NSynth y CNN para presentar

La extensión pregunta si usar vecindad local de tiempo y frecuencia mejora el mismo problema de reconstrucción y generación de sonidos. Continúa el hilo de las figuras sin comparar sus MSE con las del audio. No se cambia la fórmula del AE ni del VAE.

## Qué comparación hicimos

Se usan las1108 notas NSynth de10 familias del ejemplo, pero se reentrenan los modelos con una separación nueva por instrumento:884 notas de41 instrumentos para entrenar y224 notas de12 instrumentos diferentes para probar. Denso y CNN comparten ese split, latente16, ReLU, float32, Adam0.001, batch64 y200 épocas. Tres semillas por condición. La CNN se agrega tanto al encoder como al decoder, con filtros3×3, padding1 y stride2 para reducir resolución; el decoder usa repetición por2 y convoluciones de stride1. No hay maxpool ni convolución transpuesta. Las diferencias de parámetros de los VAE son menores al0.03%.

El VAE usa MSE+0.5 KL y calentamiento50. Para denoising se entrenan AE de MSE con ruido gaussiano sobre el mel. Este diseño permite comparar arquitecturas dentro de cada tarea, sin confundir el entrenamiento del AE con el VAE ni cambiar de representación a mitad de la comparación.

## Reconstrucción: mejora pequeña y medible

La MSE held-out pasa de0.0156975 a0.0148280: reducción media5.54%. La CNN mejora en semillas7 y 27 y empeora ligeramente en17. La gráfica muestra las tres parejas y un eje desde cero. Son tres entrenamientos independientes, no672. El resultado sugiere una ventaja modesta de la arquitectura local para reconstruir instrumentos no vistos dentro de este conjunto.

No demuestra que cada nota suene mejor. Para escuchar se ofrecen seis notas prefijadas, tres semillas y cuatro versiones: grabación original de1 segundo, referencia por el códec, VAE denso y VAE CNN. Los ejemplos de las slides son el bajo y la flauta de las posiciones0 y 2 de ese listado, no casos elegidos por mejora de la CNN.

## Generación: el resultado no se reduce a una distancia

Se decodifican los mismos64 códigos normales estándar en ambos VAE por semilla,192 salidas por arquitectura. Para respetar la medida nueva de las slides del repositorio, se calcula min MSE contra las884 notas de entrenamiento dividido por la energía media en mel de cada muestra generada. Las medianas son0.3933 en el denso y0.3683 en la CNN. Todos los casos aparecen en la gráfica.

La diferencia es pequeña. Una muestra cercana puede ser plausible, repetitiva o aproximarse a notas memorizadas. La diversidad media entre pares y el número de vecinos distintos complementan la descripción, pero no certifican novedad ni calidad. No se comparan estos valores directamente con los controles anteriores entrenados sobre todas las1108 notas.

## Denoising: mostrar también el caso negativo

Con ruido melσ0.1, la entrada tiene MSE0.007413; después del AE denso queda0.016585 y después de la CNN0.015362. Los dos empeoran lo que se quería recuperar. Conσ0.2, la entrada tiene MSE0.027086; el denso la baja a0.020476 y la CNN a0.016126: reducciones24.4%y40.5%.

Una interpretación compatible con los datos es que la reconstrucción pierde detalles que aún eran útiles cuando el deterioro era pequeño. Con deterioro fuerte, reconstruir una aproximación aprendida puede ser preferible a dejar la entrada como está. Esta interpretación no prueba un mecanismo único. El experimento es sobre corrupción de espectrogramas mel, no ruido ambiente agregado a una grabación.

## Cómo explicar las slides nuevas

Se agregan siete páginas,58–64, después de la conclusión de figuras que dice que volvemos al audio. Las69 secciones remotas se conservan con sus notas, assets y orden relativo. `presentation/index.html` sigue siendo la única presentación del repositorio.

- **58:** establece pregunta, datos y controles. Evita introducir una CNN como si el público ya conociera el protocolo de audio.
- **59:** muestra las estructuras exactas. Explica que la compresión sigue en16 valores y que cambia la forma de procesar la vecindad.
- **60:** cuantifica reconstrucción y muestra cada semilla, incluida la que empeora.
- **61:** permite comparar los mismos sonidos y distinguir red/códec.
- **62:** usa la medida relativa actual del repositorio para generación, sin convertir cercanía en prueba de novedad.
- **63:** exhibe el cruce entre dejar la entrada intacta y reconstruirla, incluyendo el fracaso con ruido moderado.
- **64:** separa lo observado en reconstrucción y denoising del objetivo aún no resuelto de notas omitidas sintéticas.

## Cambio remoto de fase y trazabilidad

Se pulleó9744710, que agrega PGHI como inicialización de fase antes de60 iteraciones Griffin-Lim. No cambia la codificación mel ni las métricas medidas en esa representación. El manifiesto de entrenamiento anterior se conserva intacto y el cambio de decodificación queda en metadatos separados. Para la página de escucha se reexportaron todos los métodos con la misma fase; el selector permite escuchar PGHI o inicialización aleatoria sin atribuir ese cambio a la CNN. Los WAV históricos no se sobrescribieron.

## Calentamiento: nuevas réplicas en curso

La comparación inicial0 vs50 épocas de calentamiento no mostró una ventaja consistente del warmup: mejora MSE sin calentamiento en3/3 semillas densas y2/3 CNN. Se agregaron semillas37,47,57 con200 épocas, idénticos datos, pesos iniciales y generadores pareados. Son12 entrenamientos nuevos que completarán seis semillas por cada combinación arquitectura/calendario; el informe distingue los resultados originales de los nuevos y no presenta datos pendientes como terminados.

No se incorpora una conclusión definitiva de esa ampliación en las slides antes de terminarla. No se busca otro beta ni otro número de épocas a partir del test.

Archivos principales: `reports/nsynth_cnn_study/`, `reports/nsynth_cnn_slides/`, `reports/nsynth_cnn_listening/index.html`, `reports/nsynth_warmup/`. La página de escucha es autónoma, sin servidor ni ZIP.
