# Bitácora de experimentos — ejercicio 2

Actualización: **22/10/2026**. Este corte incorpora desaprendizaje, confirmación y
comparación emparejada con pseudoinversa sobre los cinco QR del trabajo. Las
semillas generan nuevas corrupciones de esos mismos QR, no nuevos QR independientes.

## Estado y decisiones

| Etapa | Estado | Observación y decisión |
| --- | --- | --- |
| Prueba corta con 1369 módulos | 180 evaluaciones, completa | Menor Hamming después de desaprender; ningún original estable con los presupuestos cortos |
| Prueba corta sin 955 módulos compartidos | 180, completa | Cinco originales estables con 20 intentos; justificó ejecutar la exploración |
| Exploración completa, 1369 neuronas | 6650, completa | Ninguna combinación preservó los cinco originales en todas las semillas; no se confirmó una semilla aislada |
| Exploración reducida, 414 neuronas | 6650, completa | 10/18 combinaciones elegibles; se seleccionó aleatorio, tasa 0.1, 50 intentos |
| Confirmación reducida con semillas nuevas | 3500, completa | 90.5% medio con ruido; cinco originales estables en las tres semillas |
| Esquinas y manchas, sin reajuste | 6600, completa | Recuperación perfecta en los casos ensayados de esquinas hasta 21×21 y manchas hasta 17×17 |
| Pseudoinversa en las mismas consultas | 1400 nuevas, completa | Completa 93.3%; reducida 89.5%; desaprendizaje reducido 90.5% |

Total: **25160 evaluaciones nuevas**, incluidas las dos pruebas cortas; **24800**
sin ellas. La tabla de pseudoinversa almacena también 2100 resultados ya calculados
del candidato: esos registros reutilizados no se cuentan como evaluaciones nuevas.
El [inventario](../desaprendizaje/evidencia/inventario.csv) distingue ambas cantidades.

## 26/10 — Daño realista, sin reajustar

Se probaron siete daños que imitan un QR impreso gastado (`src/tp4/damage.py`): desgaste que solo borra tinta, desgaste mayor en un borde, manchas negras y blancas irregulares, rayones, desgaste con mancha y una foto gastada binarizada por módulo. Métodos congelados: Hebb, el candidato de desaprendizaje reducido (se reprodujo con 20/22/20 actualizaciones), pseudoinversa completa y reducida, Hamming, una variante que sabe que el daño solo borra tinta y el lector solo. Semillas nuevas 200–219. **7.800 consultas** en dos corridas; la de niveles extremos se agregó después de ver la primera.

El lector solo no lee desde 5 % de módulos alterados, ni con 2,6 % de desgaste. El desaprendizaje recuperó el 97 % o más con hasta un cuarto de los módulos alterados en los siete tipos y se rompe entre 40 y 47 %; la pseudoinversa completa aguanta más. Con daño que solo borra tinta, Hamming deja de ser el techo: la variante asimétrica acierta 97 % con 49 % alterado, contra 71 %. [Informe](../dano_realista_20261026/README.md) y [niveles extremos](../dano_realista_extremo_20261026/README.md).

La presentación del ejercicio 2 se compactó de 31 a 23 diapositivas en la rama `tp4/hopfield-compacto-dano-realista`: se agregaron el desaprendizaje y la pseudoinversa en detalle, y las dos redes pasaron a presentarse como diagnóstico (elegir la red vuelve a ser comparar con los originales).

## 22/10 — Protocolo fijado y selección

Se construye una única red con Hebb y se penalizan sólo salidas estables que no
sean recuerdos ni sus inversos. Dirigido inicia cada intento desde la mezcla
mayoritaria original; aleatorio inicia desde un vector uniforme de ±1. Ambos
conservan diagonal cero y la recuperación sincrónica. Ciclos, topes y recuerdos
protegidos consumen intentos, pero no producen updates.

Grilla por representación: dos modos, tasas 0.01/0.05/0.1, checkpoints acumulativos
10/50/100. Aleatorio usa semillas de entrenamiento 0/1/2; dirigido es determinista.
Primero se exige estabilidad de los cinco originales en todas las repeticiones.
Después se maximiza recuperación media sobre los seis niveles positivos de ruido.
La configuración seleccionada se congeló antes de evaluar semillas nuevas 100–119.

El candidato elegido tiene tasa 0.1 y 50 intentos; produjo 20/22/20 actualizaciones
efectivas en sus tres semillas. Cien intentos fue peor en exploración y no se
eligió. Esto no demuestra que 50 sea el óptimo absoluto ni que 100 sea peor ante
cualquier distribución de daño.

## 22/10 — Confirmación y comparación directa

Los cinco originales quedaron estables en las tres semillas. El desaprendizaje
reducido alcanzó **90.5%** medio sobre ruido 0.1/0.2/0.3/0.35/0.4/0.45. El
intervalo bootstrap por bloques de semillas de consulta fue 88.4–92.6%, condicionado
a estos cinco QR y las tres redes entrenadas. No se observaron ciclos o topes en
la confirmación ni en la validación espacial de ese candidato.

La pseudoinversa completa obtuvo **93.3%** en las mismas consultas. La diferencia
desaprendizaje menos pseudoinversa completa fue −2.8 puntos, intervalo −4.6 a −1.4.
Frente a pseudoinversa reducida, la diferencia fue +1.0 punto, intervalo −0.2 a
+2.3: **no concluyente**. El vecino por Hamming, comparador **sin red**, obtuvo
95.7%, por encima de las redes en promedio en esa confirmación.

**Decisión de presentación:** los resultados ya son presentables con este alcance.
Se informa la mejora frente a Hebb y la ventaja de pseudoinversa completa en esta
comparación. No se afirma superioridad universal, generalización a otros conjuntos
de QR, que desaprendizaje gane a Hamming ni que gane de forma concluyente a la
pseudoinversa reducida. Para esos claims harían falta otros conjuntos independientes
y más repeticiones de entrenamiento, no seguir ajustando sobre esta confirmación.

## Qué se publica y cómo se audita

- [Protocolo y comandos](../../../docs/experimentos_desaprendizaje.md).
- [Pruebas completas](../desaprendizaje/PRUEBAS_COMPLETAS.md).
- [Comparación con pseudoinversa](../desaprendizaje/PSEUDOINVERSA_VS_DESAPRENDIZAJE.md).
- [Resultados livianos, configuraciones efectivas y hashes](../desaprendizaje/evidencia/README.md).
- [Guía de nombres, claims y notas para presentar](../../../presentation/experimentos_hopfield_nuevos.md).

Se publican métricas y registros comprimidos, no pesos, checkpoints, matrices de
la red ni imágenes originales. La evidencia ocupa menos de 1 MB. Los directorios
locales `results/` siguen ignorados; se puede regenerar la figura y sus intervalos
usando sólo la evidencia publicada, sin cargar ni entrenar una red.

## Verificación antes de publicar

Pasaron 103 tests. Se comprobaron los hashes de cada métrica publicada, compresión sin pérdida, semillas emparejadas, ausencia de pesos/checkpoints y reproducción de la figura usando sólo evidencia publicada. El paquete de resultados y análisis ocupa aproximadamente 703 KiB; ningún archivo supera 103 KiB.
