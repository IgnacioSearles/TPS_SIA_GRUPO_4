# Revisión de la presentación nueva

Remoto revisado: `ea4ccad`. Integrado sin conflictos mediante merge `7e2b34c`.
La presentación vigente contiene **58 diapositivas**. No se modificó su HTML,
CSS, README, fuentes ni assets después de integrar lo subido por los compañeros.
Se comprobaron referencias locales: ninguna faltante o vacía. Se revisó el PDF
completo y, en tamaño grande, las nuevas páginas 39 y 47.

## Qué cambió y qué conservaría

La secuencia del ejercicio 1 ahora muestra los experimentos antes de presentar
la red elegida. La reconstrucción sin errores aparece después de esa elección.
En audio se agregó la arquitectura del AE y una comparación de 500 puntos por
fila contra la distancia entre notas reales. La explicación de las notas omitidas
ahora separa afinación, cercanía espectral y acierto conjunto.

El estilo gráfico no cambió: el CSS es el mismo. Fondo blanco, Arial, azul y
naranja, títulos que expresan un resultado, una figura dominante y una conclusión
breve al pie. Las notas contienen el detalle técnico. No trasladaría los siete
agregados anteriores automáticamente: el nuevo relato ya cubre parte de ellos.

## Prioridad 1: control del sorteo del AE

**Dónde:** después de la actual 39, «Las muestras del VAE quedan más cerca de notas reales».

**Título sugerido:** «Adaptar el sorteo también acerca el AE a notas conocidas».

**Figura:** comparación completa nueva de AE con normal estándar, el mismo AE
con una normal ajustada y VAE con normal estándar. Tres filas con espectrogramas
y audios, usando los mismos primeros índices del control complementario.
Una leyenda visible define la gaussiana como una nube de probabilidad cuyo centro
y anchura indican de dónde se sortean los números del código.

**Conclusión al pie:** «El AE conserva sus pesos: cambia la distribución usada
para generar. Eso no lo convierte en VAE».

El control disponible tiene 256 muestras por método, una semilla y reproducciones
nuevas de los modelos. Usa **medias**: 0,03213 / 0,01757 / 0,01709. La slide 39
usa **medianas de 500 muestras** y otros pesos, que no están disponibles en el
checkout. No agregar una fila a ese gráfico ni comparar ambas estadísticas como
si fueran una única corrida. Si se incluyen números, el gráfico nuevo debe contener
sus tres métodos y declarar claramente su procedencia. No hace falta reentrenar.

## Prioridad 2: alcance de la evidencia de generación

**Dónde:** después de la actual 43, «El VAE genera notas nuevas, no copias del dataset».

**Título sugerido:** «Parecerse a una nota no demuestra que sea nueva».

**Figura:** tres preguntas breves: si tiene estructura de nota, si es idéntica a
alguna referencia y si generaliza a una condición no vista. A cada una se asocia
su prueba: escucha/espectrograma, comparación con vecinos y experimento reservado.
No introducir un nuevo puntaje combinado ni afirmar que la escucha ya fue validada
por un estudio humano.

**Conclusión al pie:** «Las distancias muestran parecido y diferencias; no
certifican calidad auditiva, novedad ni ausencia de memorización».

Es una aclaración adicional, sin reescribir el título ni las notas existentes.
Una muestra puede ser una reproducción ligeramente alterada; una nota genuinamente
distinta también puede quedar cerca del dataset. Alcanzar la distancia entre dos
notas reales no implica por sí solo copiar. Asimismo, 500 muestras de una red no
son 500 entrenamientos independientes. Este último detalle puede quedar en las notas.

## Aclaración necesaria del protocolo de omisiones

**Dónde:** como página de respaldo inmediatamente después de las actuales 56–57,
o junto a la 47 si hay tiempo para explicar el experimento en el cuerpo principal.

**Título sugerido:** «Excluir notas y elegir extremos son decisiones distintas».

**Figura:** línea de alturas alrededor de un objetivo. En rojo se marca la zona
excluida del entrenamiento; en azul se señalan los dos extremos que se codifican.
Comparar omisión normal y estricta sin confundir ambas decisiones.

El código de `src/evaluation/pitch_interpolation.py` codifica siempre
`target - 1` y `target + 1`. En el dataset microtonal la exclusión normal de radio
0,5 deja como notas de entrenamiento más cercanas las de ±0,75; la estricta de
radio 0,75 deja las de ±1. Por eso las etiquetas nuevas «vecinas a ¾» y
«vecinas a 1» describen el entorno disponible de entrenamiento, no un cambio de
extremos de interpolación. El punto medio sigue usando los vecinos enteros.

**Conclusión al pie:** «Cambia el hueco del entrenamiento; los extremos de la
interpolación siguen siendo objetivo −1 y objetivo +1».

Esto no altera los recuentos 5/27 y 2/27 ni la conclusión negativa. Es una
precisión sobre qué se comparó, no un resultado adicional. No se modificó la 47.

## Extensión visual, si quieren incorporar las figuras

La incluiría como **dos páginas consecutivas** después de la actual 47 y antes
de «Conclusiones», claramente identificadas como extensión visual. Si el tiempo
es limitado, ambas pueden ir al apéndice. No mostraría GMM sin este contexto.

1. **«Reconstruir una figura no garantiza generarla bien».** Una referencia real
   con su reconstrucción, frente a cuatro generaciones sin imagen de entrada.
   Rótulos grandes: «Con entrada» / «Sin entrada». Al pie: «Mismo mecanismo
   variacional que en audio; otro modelo, otros datos y otras métricas».
2. **«Una imagen limpia puede estar mal dibujada».** Los dos fallos reales de la
   semilla 27, índices 13 y 61, ampliados y señalados. Definiciones visibles:
   limpieza = fondo uniforme, color parejo y una región principal; forma correcta
   = contorno reconocible como alguna de las cuatro familias. Al pie:
   «Como en audio: acertar la altura no garantiza conservar los armónicos».

Las generaciones son no condicionales: no tienen una clase objetivo asignada.
No marcar una como «círculo incorrecto» sin esa aclaración. Los dos ejemplos
ilustran fallos; no estiman su frecuencia. No comparar MSE de RGB con MSE de audio.

## GMM: solo respaldo opcional

**Dónde:** después del bloque RGB, en el apéndice.

**Título:** «Varias nubes pueden describir mejor dónde están los códigos».
Mostrar una nube global frente a varias, con la definición visible «GMM = mezcla
de gaussianas». Aclarar que son esquemas, no una proyección medida ni clases.
El resultado de 93,5 % a 98,8 % se refiere a limpieza, no a figuras correctas;
se mantienen los decoders y cambia la distribución de generación. No es una
mejora del VAE estándar por reentrenamiento ni una prueba de novedad.

## Qué no agregaría todavía

- Resultados de la CNN de audio: la primera fase aún está entrenando; no hay
  corridas principales terminadas ni evaluación comparativa. No anticipar mejoras.
- Otra explicación completa del VAE: las páginas 34–35 ya cubren arquitectura
  y pérdida. Una breve aclaración AE/gaussiana basta junto al nuevo control.
- Grillas de 32 mapas de activación en el cuerpo: requieren explicación adicional
  y no responden una pregunta que falte en la presentación nueva.
- Una tabla con todo el historial de búsquedas o el piloto de beta: aporta poco
  al hilo principal y alarga una presentación que ya tiene 58 páginas.

## Aplicación del estilo

Conservar 1280×720, Arial, tamaños y paleta actuales. Un mensaje por página,
evidencia central grande, referencias pegadas a la parte relevante y una sola
conclusión al pie. Las definiciones necesarias para entender la figura deben
ser visibles; las semillas, parámetros, limitaciones y procedencia van en notas.
No usar paneles decorativos, tablas densas ni nuevas escalas gráficas sin motivo.

La numeración de esta propuesta es la vigente **antes** de insertar páginas.
Para implementarla se debe partir del nuevo `presentation/index.html`, no de
los builders anteriores fijados a una base de 55 diapositivas. Todo agregado
debe comprobar que las 58 secciones actuales se conservan literalmente y en orden.
