# Agregados sobre la versión nueva: valor, evidencia y límites

Base: `7e2b34c`, que integra el remoto `ea4ccad`. Son **58 páginas originales
conservadas literalmente**, más seis agregados: **64 páginas** en total.
El archivo original `presentation/index.html`, su CSS, notas y assets no se editan.
La versión adicional es `presentation/con_agregados_actualizados.html`.

| Página nueva | Ubicación en la versión original | Qué aporta |
|---|---|---|
| 40 | Después de la 39, distancias a notas | Separa el efecto del entrenamiento del efecto de elegir otra distribución para generar. |
| 45 | Después de la 43, muestras nuevas | Distingue diferencia observable de novedad demostrada y evita interpretar una distancia como prueba de no memorización. |
| 50 | Después de la 47, notas omitidas | Extiende la misma pregunta de reconstrucción/generación a una modalidad que se puede inspeccionar visualmente. |
| 51 | Después de la extensión RGB | Muestra fallos que la métrica de limpieza no detecta y conecta con afinación frente a estructura espectral. |
| 62 | Apéndice, tras pérdida auxiliar | Aclara que cambian las alturas disponibles en entrenamiento, pero los extremos de interpolación siguen siendo ±1. |
| 64 | Apéndice, tras la última original | Presenta la mejora de limpieza con GMM y deja visible que eso no garantiza geometría correcta. |

## Qué se graficó y por qué

### 40: sorteo del AE

Gráfico nuevo con **las 256 distancias individuales de cada uno de tres métodos**,
sin recortar muestras: AE con normal estándar, el mismo AE con gaussiana ajustada,
y VAE con normal estándar. Las líneas marcan medias. El AE ajustado reduce la
distancia media a entrenamiento un 45,3 %: 0,03213 a 0,01757; el VAE queda en 0,01709.

La definición visible de gaussiana explica centro y anchura de la nube. La conclusión
aclara que el AE conserva sus pesos y no se convierte en VAE. Es un control con
una semilla y reproducciones nuevas: **no comparte pesos ni el estadístico del
gráfico remoto de 500 puntos y medianas**. No se añadió una fila a aquel gráfico.
Los datos por muestra se exportan a `sorteo_256.csv`.

### 45: diferencia frente a novedad

Se usa la primera muestra del VAE del control nuevo, su vecino más próximo entre
todas las notas de entrenamiento y el mapa de diferencia absoluta. Se comprueba
que la MSE del par coincida con el CSV guardado. Las escalas de ambos espectrogramas
son iguales; la diferencia tiene barra propia. No es un montaje de audios distintos.

Permite mostrar qué cambió en la representación, sin afirmar que la muestra sea
nueva, musicalmente mejor o imposible de obtener por una variación memorizada.
La identidad del vecino se registra en `ejemplo_distancia.json`.

### 50–51: estructura visual

El rectángulo de validación índice 3 y su reconstrucción dejan ver el redondeo
de esquinas. Cuatro generaciones desde el prior usan índices 0,16,32,48 de la
semilla 7, una secuencia equiespaciada; no se seleccionaron las mejores.

Los fallos de la página 51 son los índices 13 y 61 de la semilla 27. Se amplían
y se señalan una curva asimétrica y un lado hundido. Se eligieron como **casos
ilustrativos de deformación**, no para estimar su frecuencia. Ambos pasan limpieza.
No les asignamos una clase objetivo que la generación libre no tiene.

Las definiciones de limpieza y forma están visibles. Los detalles de umbrales
quedan en notas. La conexión con audio es conceptual: conservar una propiedad
no garantiza conservar toda la estructura. No comparamos MSE de modalidades
distintas ni suponemos pesos o espacios latentes compartidos.

### 62: esquema del protocolo

La línea representa los microtonos realmente disponibles. Cruces naranjas marcan
alturas excluidas; azul marca los extremos que se codifican. La exclusión normal
de radio 0,5 deja las notas entrenadas más próximas a ±0,75; la estricta de radio
0,75 las deja a ±1. El interpolador usa siempre `target−1` y `target+1`.

Esto precisa las etiquetas de la página remota 47 sin modificarla y sin cambiar
los resultados 5/27 o 2/27. No se presenta como un nuevo experimento.

### 64: GMM

Gráfico de proporciones con eje completo 0–100, sin ampliar visualmente el pequeño
intervalo 90–100. Se muestran recuentos y porcentajes de las 768 imágenes por método:
normal estándar 718, GMM1 738, GMM4 756 y GMM8 759. Se conservan los tres decoders.
La definición de GMM —mezcla de gaussianas— es visible.

Cuatro ejemplos de GMM8 usan 0,64,128,192: una de cada dos posiciones del recorrido
prefijado, cubriendo el rango de 256 salidas. No son los mejores ejemplos ni un
antes/después de las otras imágenes. La conclusión distingue claramente 98,8 %
de limpieza de un supuesto 98,8 % de formas correctas. GMM4 y GMM8 difieren en
solo tres imágenes limpias sobre 768; no se proclama superioridad universal.

## Experimentos CNN de audio

Se leyó el estado de los checkpoints, sin interrumpir ni cambiar entrenamientos.
La primera fase tiene los tres CNN con notas conocidas en entrenamiento. Al preparar
este material **todavía no había corridas principales terminadas**.

Su valor potencial es aislar arquitectura: mismos datos por condición, códec,
latente, MSE+KL, épocas y unas cantidades de parámetros casi iguales. Hasta completar
controles, omisiones y sus evaluaciones no aporta una mejora musical demostrada.

`estado_audio.json` registra las épocas efectivamente guardadas. `avance_audio.png`
muestra MSE de entrenamiento por época y semilla, con escala logarítmica. Se deja
**fuera de las slides de resultados**: no es evaluación de notas reservadas ni
comparación terminada contra el VAE denso. Que esa curva baje no demuestra éxito.

## Preservación y reproducción

`verification.json` contiene hashes de todos los archivos previos de la presentación,
el commit base, las posiciones nuevas y los índices de los casos. El builder comprueba
que quitar las seis inserciones y el vínculo al CSS nuevo reconstruya exactamente
el HTML original de 58 páginas, y que ningún archivo previo haya cambiado.

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 && source .venv/bin/activate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.build_presentation_additions_current
```

No se ejecutó `sync_assets.py` ni se regeneraron assets sobre rutas existentes.
Los gráficos nuevos usan el prefijo `extra_actualizado_`. Solo se añadió un CSS
con reglas acotadas a las secciones nuevas. El estilo mantiene fondo blanco,
Arial, paleta azul/naranja, título de hallazgo, figura dominante y conclusión al pie.
