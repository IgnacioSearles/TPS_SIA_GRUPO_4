# Guion final: qué agregar, por qué y cómo explicarlo

## Alcance y regla vigente

El estudio se cerró el 15 de noviembre de 2026. La presentación existente no se
reescribe, reordena ni recorta. Todo lo siguiente es material adicional, con
secciones y assets nuevos; las notas actuales también se conservan.
La propuesta anterior de reemplazar una comparación queda descartada.

Recomiendo **tres diapositivas principales nuevas y dos de apéndice**. Los lugares
se indican por el título o tema de la página existente, para no depender de una
numeración que cambiará cuando se agreguen secciones. No agregaría el historial
completo de pruebas al cuerpo de la exposición.

## Nueva 1: «La distribución de muestreo también importa»

**Dónde:** después de la comparación actual AE/VAE desde N(0,I).

**Qué mostrar:** una comparación nueva completa, con los tres métodos del control
de NSynth. Usar los tres primeros índices, iguales entre métodos, y sus WAV.
No agregar una fila de otra corrida debajo de las imágenes originales: esta
página adicional debe tener sus propios ejemplos y metadatos. Los pesos de
la presentación original no estaban en el checkout; son reproducciones nuevas,
con la misma configuración, no los mismos pesos certificados.

**Texto visible:**

- AE desde normal estándar: MSE al vecino más próximo, 0,0321.
- Mismo AE desde gaussiana ajustada: 0,0176.
- VAE desde normal estándar: 0,0171.

Debajo, una línea: «Cercanía al entrenamiento; no calidad auditiva ni novedad».

**Por qué aporta:** la comparación original permite estudiar muestreo estándar,
pero no descarta que un AE genere usando una distribución adaptada a sus códigos.
Este control separa la elección de los puntos del efecto de entrenar un VAE.
Es también una respuesta concreta a la pregunta de si un AE con gaussiana es VAE.

**Guion oral, unos 50 segundos:**

> Para generar necesitamos elegir un código sin una entrada. En el VAE usamos
> una normal estándar porque su entrenamiento regulariza los posteriores hacia
> ese prior. En el AE nadie impuso dónde debían quedar los códigos. Por eso
> probamos otra alternativa: ajustar una gaussiana a los códigos de entrenamiento,
> conservando los mismos pesos. Su cercanía al conjunto conocido mejoró un
> 45,3 %. Eso muestra que parte de la comparación dependía del muestreo.
> El AE sigue siendo AE: no aprendió media y varianza por entrada ni recibió KL.
> Estas distancias no nos dicen por sí solas si el sonido es bueno o nuevo.

**Si preguntan por la matemática:** el AE ajustado usa z=m+L·epsilon, con
media y covarianza globales obtenidas después de entrenar. El VAE aprende
q(z|x) por entrada y, durante su entrenamiento, usa
z=mu(x)+sigma(x)·epsilon y penaliza KL. Son procedimientos distintos.

**Archivos nuevos disponibles:** `figuras/audio_muestreo.png` y
`audio/{ae_standard,ae_fitted,vae_standard}_{0,1,2}.wav`. Los espectrogramas se
inspeccionaron; no se realizó una evaluación auditiva humana controlada.

## Nueva 2: «El mismo mecanismo variacional con figuras RGB»

**Dónde:** después del estudio de notas omitidas y antes de las conclusiones
actuales. Introducirla como una extensión visual del ejercicio 2.

**Qué mostrar:** originales de validación y reconstrucciones, junto con ocho
generaciones desde N(0,I). Usar `figuras/rgb_reconstruccion.png` y
`figuras/rgb_generaciones.png`: referencias 0–3 y generaciones 0,8,…,56 de la
semilla 7. Son índices prefijados, no los mejores resultados elegidos al final.

**Texto visible:**

- Cuatro familias y variaciones de tamaño, orientación y color.
- Píxeles RGB de 32×32, latente de 8 dimensiones.
- Encoder, posterior y decoder aprendidos; sin etiquetas geométricas como entrada.

**Por qué aporta:** permite ver inmediatamente qué se conserva o se pierde,
complementando la escucha y los espectrogramas. Mantiene la pregunta común:
aprender una representación para reconstruir y generar ejemplos plausibles.
No requiere afirmar que las figuras y los sonidos sean tareas idénticas.

**Guion oral, unos 45 segundos:**

> Para observar el mismo problema con una salida que podemos reconocer a simple
> vista usamos figuras de colores. El modelo recibe píxeles, sin que le pasemos
> el nombre de la figura o sus medidas. Arriba están las reconstrucciones de
> imágenes reservadas y abajo ejemplos generados sin una imagen de entrada.
> Compartimos el mecanismo variacional del audio, pero entrenamos otro modelo.
> No compartimos pesos ni un espacio latente. Tampoco comparamos sus MSE:
> el error RGB y el error de un espectrograma están definidos sobre datos distintos.

**Para preguntas:** son 3.072 valores RGB y un vector z de 8 coordenadas.
La reducción dimensional respecto de z es 384 a 1; no es una tasa de compresión
de archivos, porque omite precisión, parámetros del posterior y pesos del modelo.
El VAE de audio usa latente 16 y beta 0,5; el RGB refinado, latente 8 y beta final
0,02, después de fases previas. No se está aislando causalmente el efecto de la modalidad.

## Nueva 3: «Una imagen limpia puede tener una forma incorrecta»

**Dónde:** después de la nueva página RGB. Esta es la conclusión visual principal.

**Qué mostrar:** los dos casos de `figuras/limpia_pero_deforme.png`, que pasan
limpieza pero tienen contornos irregulares, y una tabla pequeña del control
pareado: un bloque, 91,1 %; dos bloques, 94,8 %. La auditoría confirma una
mejora más modesta, de 90,4 a 93,2 %. Ambas comparaciones usan todos los casos.
Los dos fallos son ilustraciones explícitas, semilla 27 e índices 13 y 61;
no se usan para estimar una tasa de fallos geométricos.

**Texto visible:**

- El refinamiento es una convolución aprendida dentro del decoder.
- Limpieza mejora; IoU media de reconstrucción casi igual, alrededor de 0,934.
- Limpieza automática y figuras inequívocas son propiedades diferentes.

**Por qué aporta:** explica qué produjo la mejora y evita equiparar una cifra
alta con calidad visual total. También muestra una evaluación que reconoce
sus límites y conserva los casos negativos.

**Guion oral, aproximadamente un minuto:**

> Agregamos una convolución antes de producir RGB. Combina mapas de posiciones
> vecinas y aprende con la misma pérdida; no tiene instrucciones para dibujar
> círculos o triángulos. La insertamos como identidad, de modo que inicialmente
> conservaba la función del modelo. Con los mismos padres, datos y pasos,
> mejoró la limpieza en las tres semillas. Pero estos dos ejemplos pasan
> limpieza y están deformados. La métrica revisa fondo, color y bordes,
> no exige tres lados o cuatro ángulos. Nuestro criterio visual completo no
> se cumplió, por lo que no declaramos éxito definitivo ni abrimos la prueba final.

**Qué explica las deformaciones:** la MSE castiga diferencias de píxeles, no
reglas geométricas. Con incertidumbre puede favorecer promedios que suavicen
contornos. La resolución pequeña limita las esquinas; también pueden aparecer
intermedios entre familias en un espacio continuo. El muestreo puede visitar
regiones distintas de las mejor representadas durante el entrenamiento.
Son explicaciones compatibles con la evidencia, no causas individuales demostradas.

**Si preguntan por la capa:** es una convolución 3×3 de 16 a 16 canales,
que suma 2.320 parámetros. Arranca copiando los canales desde el centro del
kernel y luego aprende todos sus coeficientes. Forma parte del decoder,
incluida la generación desde ruido. Un VAE tradicional puede ser convolucional.
Se conservan MSE, KL y reparametrización; cambia la arquitectura.

## Apéndice nuevo A: «Dónde elegimos los códigos»

**Qué mostrar:** `figuras/comparacion_final.png`, o las comparaciones completas
por semilla, y la tabla del último experimento:

| Distribución para generar RGB | Limpias, tres modelos |
|---|---:|
| Prior N(0,I) | 718/768: 93,5 % |
| GMM de 1 componente | 738/768: 96,1 % |
| GMM de 4 componentes | 756/768: 98,4 % |
| GMM de 8 componentes | 759/768: 98,8 % |

**Explicación:** las GMM ajustan unas pocas regiones de la distribución de
códigos de entrenamiento, sin usar etiquetas. Mantienen el decoder y cambian
la distribución de generación. Es un procedimiento de dos etapas: VAE más
densidad ajustada. Las componentes no tienen por qué corresponder a familias.

Los cuatro y ocho componentes dan resultados próximos. La GMM de ocho mejora
limpieza frente al prior en este conjunto, pero todavía produce figuras híbridas.
No se presenta como generación estándar desde N(0,I) ni como prueba de novedad.
Las filas de las imágenes tampoco son el mismo objeto antes y después.

**Referencia diagnóstica adicional:** muestrear la mezcla de los 4.096
posteriores de entrenamiento dio 99,9 % de limpieza, pero quedó muy cerca de
reconstruir imágenes conocidas. Sirve para observar la capacidad del decoder,
no para proclamar muestras novedosas. Una GMM pequeña evita guardar un centro
por imagen, aunque tampoco descarta memorización en la red.

## Apéndice nuevo B: «Controles y límites»

Mostrar tres semillas para RGB, una para el control nuevo de NSynth, fuentes,
datos usados en el ajuste y generaciones con semillas prefijadas. Separar las
referencias de entrenamiento, validación y la prueba final que quedó sin abrir.

Añadir el piloto negativo de beta: 20 épocas pareadas, beta 0,02 o 0,1,
62/64 limpias y 235/256 en ambas ramas. La KL bajó de 28,92 a 22,32 sin
mejorar esas generaciones. Esto no prueba que una KL mayor nunca ayude.

La [nota de regularización](../ej2_search/interpretacion_regularizacion.md)
incluye la descomposición teórica de la KL y una fuente primaria para preguntas.

Las barras de error entre notas o generaciones de un único modelo no miden
variabilidad entre entrenamientos. La cercanía al dataset no es novedad ni
calidad perceptual. Una proyección 2D o momentos parecidos a N(0,I) tampoco
establecen igualdad de densidades en todas las dimensiones. Estas aclaraciones
se agregan en una página nueva, conservando los textos y notas actuales.

## Preguntas que conviene preparar

**¿Un AE con gaussiana ajustada es un VAE?** No. Ajusta una densidad global a
códigos deterministas después de entrenar. No aprende incertidumbre por entrada
ni recibió la regularización KL del VAE.

**¿La GMM arregló el VAE estándar?** Mejoró limpieza al usar otra distribución
de generación. El VAE estándar desde N(0,I) no cambió. Es un control y una
variante exploratoria, no una aprobación retroactiva del criterio original.

**¿Por qué no bastó subir beta?** La KL por entrada penaliza tanto información
como desajuste con el prior. Una penalización mayor puede perder detalle y no
mejorar la generación; eso fue lo observado en el piloto corto, no una ley universal.

**¿Se demostró que las muestras no están memorizadas?** No. Se guardan vecinos,
distancias y diversidad para diagnosticar, pero esas medidas no lo certifican.

**¿Por qué quedan figuras malas si la reconstrucción es buena?** Reconstruir
elige z mediante una entrada; generar lo elige mediante una distribución.
Son dos pruebas diferentes. Además la IoU media puede ocultar algunos fallos.

**¿Contradice el resultado negativo de notas omitidas?** No. Generar ejemplos
plausibles o interpolar timbres es una exigencia distinta de inferir una altura
que se eliminó del entrenamiento. No se declara que aprendió una regla musical.

**¿Qué relación tienen sonidos y figuras?** Comparten el problema de aprender
una representación y generar. En audio importa conservar estructura espectral;
en imágenes, la estructura geométrica. Se evalúa cada propiedad con sus controles,
sin postular pesos compartidos ni comparar sus escalas de error.

## Frase de cierre sugerida para una página nueva

> La representación y la distribución de muestreo influyen en lo que generamos.
> El VAE favorece un prior utilizable, pero la calidad depende del aprendizaje
> efectivo y de la capacidad del decoder. Las métricas deben medir la propiedad
> que queremos conservar: acertar una altura no alcanza para conservar un espectro,
> y limpiar una imagen no alcanza para conservar una figura.

Todo asset se incorporaría con un nombre nuevo, por ejemplo bajo un prefijo
`extra_cierre_vae_`. El material está listo para agregarlo; esta tarea no modificó
ni eliminó páginas, notas, estilos o assets de la presentación existente.
