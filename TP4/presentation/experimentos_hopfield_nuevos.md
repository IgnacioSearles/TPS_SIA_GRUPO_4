# Nuevos experimentos de Hopfield: guía para presentar

Estas notas complementan la presentación existente. No cambian las diapositivas
ni sus resultados históricos. [Bitácora](../reports/ejercicio2/bitacora/README.md).

## Mensajes que la evidencia sostiene

### 1. Desaprender espurios mejora la red hebbiana reducida

Mostrar tres cifras: **Hebb 0%**, **Hebb con desaprendizaje 90.5%**, **Hamming sin
red 95.7%**. Son medias sobre los seis niveles positivos de ruido, con consultas
de confirmación nuevas y los mismos cinco QR.

Notas: una única red de 414 neuronas recupera los módulos variables y se reponen
955 módulos conocidos. Las tres semillas de entrenamiento son repeticiones; no
se combinan las salidas ni forman un ensamble. Cada repetición conservó estables
los cinco originales. El comparador por Hamming selecciona directamente un original
y no tiene recuperación de Hopfield.

### 2. Más desaprendizaje no garantizó mejores resultados

Mostrar **50 intentos: 88.4%** y **100 intentos: 86.0%**, ambos con tasa 0.1 y
representación reducida, en exploración. Se seleccionó el primero y luego se
confirmó con ruido nuevo. No presentar 50 como óptimo universal ni esos porcentajes
exploratorios como confirmación de ambos presupuestos.

Notas: 50 intentos dieron 20/22/20 actualizaciones efectivas. Los otros intentos
terminaron en estados protegidos. El entrenamiento no se detiene automáticamente
por alcanzar precisión ni se sigue modificando la red durante la evaluación.

### 3. Pseudoinversa completa fue la mejor de las redes comparadas aquí

Mostrar **pseudoinversa completa 93.3%**, **desaprendizaje reducido 90.5%** y
**pseudoinversa reducida 89.5%**, sobre las mismas consultas nuevas. La diferencia
de un punto entre las dos reducidas no es concluyente. Hamming sin red sigue por
encima en promedio, con 95.7%.

Notas: no comparar tablas antiguas con semillas distintas para sostener esta
conclusión; usar [la comparación emparejada nueva](../reports/ejercicio2/desaprendizaje/PSEUDOINVERSA_VS_DESAPRENDIZAJE.md).
Que los comunes no distingan la identidad no implica que no modifiquen la dinámica.
La representación completa y la reducida tienen matrices de pesos diferentes.

### 4. El daño localizado también se evaluó sin reajustar el candidato

Mostrar esquinas blancas hasta **21×21** y manchas negras hasta **17×17** con
100% en los casos ensayados. Después, mostrar al menos un tamaño que falle:
esquinas 29×29, 75%; manchas 29×29, 91%.

Notas: tamaño cubierto no equivale a cantidad de módulos invertidos. Los comunes
se reponen con información conocida. Esta tanda mide recuperación de módulos;
no se corrió un lector óptico y no se demuestra robustez de detección fotográfica.

### 5. Con daño realista, el problema práctico es el lector (26/10)

Mostrar que el lector solo no lee desde **5 %** de módulos alterados y que el
desaprendizaje recupera el **97 % o más** con hasta un cuarto de los módulos
alterados, en desgaste, manchas, rayones y foto gastada. Diapositiva 57;
[informe](../reports/ejercicio2/dano_realista_20261026/README.md).

Notas: son simulaciones con métodos congelados y semillas nuevas. La corrida de
niveles extremos se agregó después de ver la primera. Con daño que solo borra
tinta, Hamming deja de ser el techo.

### 6. Dos redes es un diagnóstico, no una solución

Con la consulta en la red que guarda su QR, 100 % hasta 35 %: confirma que el
par menú-vinos es el obstáculo. Pero elegir la red vuelve a ser comparar con los
originales (desempate por Hamming) o algo equivalente (energía con Hebb). No la
presentamos entre las soluciones. Diapositiva 47.

## Nombres exactos para las leyendas

- **Hopfield — Hebb (caso base)**.
- **Hopfield — pseudoinversa completa, 1369 neuronas**.
- **Hopfield — pseudoinversa reducida + restauración de compartidos**.
- **Hopfield — Hebb con desaprendizaje aleatorio reducido + restauración**.
- **Vecino más cercano por Hamming — referencia sin red**.
- Si se muestran los experimentos anteriores de varias redes: **redes de Hopfield
  con Hebb + selector externo por Hamming**. No confundir con la referencia sin red.

## Límites que deben quedar en las notas

Los resultados se condicionan a estos cinco QR. Semillas de ruido nuevas no
equivalen a nuevos QR. Los intervalos remuestrean semillas de consulta como
bloques, manteniendo fijo el conjunto de QR y las tres redes entrenadas. No hay
evidencia para afirmar generalización a otros menús o que una regla gane siempre.

La red completa con desaprendizaje no pasó el requisito de estabilidad en todas
las semillas: se conserva ese resultado negativo. En algunas configuraciones
agresivas aparecieron ciclos; no afirmar que toda actualización sincrónica reduce
energía ni que debilitar la mezcla garantiza recuperar los originales.

## Fuentes listas para utilizar

- [Informe completo y gráfico de confirmación](../reports/ejercicio2/desaprendizaje/PRUEBAS_COMPLETAS.md).
- [Comparación entre reglas](../reports/ejercicio2/desaprendizaje/PSEUDOINVERSA_VS_DESAPRENDIZAJE.md).
- [Evidencia comprimida, sin pesos ni checkpoints](../reports/ejercicio2/desaprendizaje/evidencia/README.md).

La historia está lista para presentar con estos límites. Para ampliar los claims,
el siguiente paso es otra colección independiente de QR; no volver a seleccionar
parámetros después de mirar las consultas de confirmación de esta tanda.
