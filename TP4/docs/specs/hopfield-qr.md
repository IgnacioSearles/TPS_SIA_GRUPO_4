# Spec — Ej. 2: recuperación de códigos QR deteriorados con Hopfield

> **Cambio posterior (2026-10-20):** a pedido de Ivo se quitó el modo asincrónico (el enunciado no lo pide). La red, el análisis y el informe usan sólo la actualización sincrónica del apunte. Quedan sin efecto las historias 7, 9, 11, 30 y 36, la parte asincrónica de 13, 31–33, 71, 74–75, 78, 81 y de la sección de tests, y la nota sobre los dos modos de los antecedentes.

Estado: lista para implementar (2026-10-20). Vocabulario: `GLOSSARY.md`. Decisión previa: ADR 0001 (sólo regla de Hebb).

## Problem Statement

El grupo tiene que entregar el ejercicio 2 del TP4: usar una red de Hopfield como memoria asociativa para recuperar los cinco códigos QR de la Trattoria Aurora (menú principal, vinos, cócteles y otras bebidas, postres, especiales del día) a partir de versiones deterioradas. La consigna pide representar cada QR como una matriz binaria de 37×37, almacenar los cinco, construir patrones de consulta con distintos niveles de ruido, analizar las iteraciones y la energía en cada paso, y explicar las limitaciones del caso de uso.

No hay nada implementado del ejercicio 2. Una medición previa mostró además que, con la regla de Hebb del apunte, ninguno de los cinco QR es un estado estable: los cinco caen al mismo estado mezcla. El trabajo no puede presentarse como "la red recupera los QR"; tiene que mostrar con evidencia qué hace la red, por qué, y qué implica para el restaurante.

## Solution

Una red de Hopfield implementada desde cero con NumPy, fiel a las fórmulas del apunte de la Unidad 9, que almacena los cinco QR con la regla de Hebb y recupera patrones de consulta en dos modos de actualización: sincrónico (el visto en clase, modo principal) y asincrónico (como comparación, con las mismas consultas). Cada recuperación deja registro de su estado final, sus iteraciones, la energía en cada iteración, el motivo por el que se detuvo y su categoría de resultado.

Sobre esa red, un análisis reproducible de los QR que primero estudia la ortogonalidad de los patrones almacenados (y anticipa el colapso), después corre 1100 patrones de consulta en ambos modos, y como experimento secundario verifica con un lector de QR real si el original, la consulta y la salida de la red se pueden decodificar. Todo termina en un informe del ejercicio 2 con figuras, tablas e interpretación, cuya sección central es la de limitaciones.

## User Stories

1. As an integrante del grupo, I want cargar cada QR original como un patrón de ±1 de 1369 módulos, so that la red trabaje directamente con los datos de la consigna.
2. As an integrante del grupo, I want que negro sea +1 y blanco −1 en todos los patrones, so that las figuras y las cuentas usen una única convención.
3. As an integrante del grupo, I want que la carga falle con un mensaje claro si una imagen no es de 37×37 o no es binaria, so that no se analicen datos mal representados sin darme cuenta.
4. As an integrante del grupo, I want almacenar los cinco QR en la red con la regla de Hebb del apunte, so that el modelo sea el visto en clase.
5. As an integrante del grupo, I want que la matriz de pesos sea simétrica y con diagonal cero, so that cumpla las condiciones del modelo de Hopfield.
6. As an integrante del grupo, I want presentarle a la red un patrón de consulta y obtener el estado al que llega, so that pueda ver con qué patrón almacenado lo asocia.
7. As an integrante del grupo, I want elegir el modo de actualización (sincrónico o asincrónico) en cada recuperación, so that pueda comparar los dos sobre la misma red.
8. As an integrante del grupo, I want que en modo sincrónico todas las neuronas se recalculen a la vez con sign(W·S), so that coincida con el algoritmo del apunte.
9. As an integrante del grupo, I want que en modo asincrónico cada iteración recorra las N neuronas en un orden aleatorio nuevo, so that "iteración" signifique lo mismo en los dos modos.
10. As an integrante del grupo, I want que una neurona con campo exactamente cero conserve su estado, so that se respete la regla del apunte.
11. As an integrante del grupo, I want fijar una semilla para el orden asincrónico, so that una recuperación se pueda repetir y dé el mismo resultado.
12. As an integrante del grupo, I want que la recuperación se detenga al llegar a un estado estable, so that no se hagan iteraciones de más.
13. As an integrante del grupo, I want que la recuperación se detenga si vuelve a un estado ya visitado, marcándola como ciclo, so that las oscilaciones del modo sincrónico queden identificadas y no se confundan con un resultado.
14. As an integrante del grupo, I want un tope de 100 iteraciones, so that ninguna recuperación quede corriendo indefinidamente.
15. As an integrante del grupo, I want saber por qué se detuvo cada recuperación (estado estable, ciclo o tope), so that pueda reportar cuántas no llegaron a un estado estable.
16. As an integrante del grupo, I want la energía del estado inicial y la de cada iteración, so that pueda analizar la energía en cada paso como pide la consigna.
17. As an integrante del grupo, I want la cantidad de iteraciones de cada recuperación, so that pueda analizar cuánto tarda la red en estabilizarse.
18. As an integrante del grupo, I want que cada recuperación se clasifique como recuperación correcta, otro patrón almacenado, patrón inverso o estado espurio, so that pueda resumir los resultados sin mirar cada salida.
19. As an integrante del grupo, I want que la clasificación se haga respecto del patrón almacenado del que salió la consulta, so that "correcta" signifique volver al QR original.
20. As an integrante del grupo, I want la distancia de Hamming entre el estado final y el QR original, so that pueda medir qué tan lejos quedó la red cuando no lo recupera.
21. As an integrante del grupo, I want la matriz de solapamientos entre los cinco QR, so that pueda analizar la ortogonalidad antes de almacenarlos.
22. As an integrante del grupo, I want contar cuántos módulos son iguales en los cinco QR, so that pueda explicar por qué QR que se ven distintos son casi iguales para la red.
23. As an integrante del grupo, I want medir el crosstalk de cada QR almacenado, so that pueda mostrar con la fórmula del apunte por qué no quedan estables.
24. As an integrante del grupo, I want comparar la cantidad de QR con la capacidad del apunte (0,15·N ≈ 205), so that quede claro que la falla no es por cantidad sino por correlación.
25. As an integrante del grupo, I want verificar si cada QR sin ruido es un estado estable, so that el informe parta de ese hecho.
26. As an integrante del grupo, I want verificar si el estado al que caen los QR es el estado mezcla sign(ξ₁ + … + ξ₅), so that el estado espurio quede explicado y no sólo observado.
27. As an integrante del grupo, I want construir patrones de consulta invirtiendo cada módulo con probabilidad p, so that se simule el deterioro como pide la consigna.
28. As an integrante del grupo, I want niveles de ruido p de 0 a 0,5 en pasos de 0,05, so that la curva cubra desde el QR limpio hasta el ruido máximo con sentido.
29. As an integrante del grupo, I want 20 semillas de ruido por nivel y por QR, so that los resultados no dependan de una única consulta.
30. As an integrante del grupo, I want que los dos modos de actualización reciban exactamente los mismos patrones de consulta, so that la comparación sea pareja.
31. As an integrante del grupo, I want una tabla con una fila por recuperación (QR, nivel de ruido, semilla, modo, categoría, Hamming, iteraciones, motivo de parada), so that cualquier figura se pueda rehacer desde los datos.
32. As an integrante del grupo, I want una figura de la distribución de categorías según el nivel de ruido para cada modo, so that se vea adónde cae la red.
33. As an integrante del grupo, I want una figura de las iteraciones según el nivel de ruido para cada modo, so that se vea cuánto tarda en estabilizarse.
34. As an integrante del grupo, I want curvas de energía por iteración para ejemplos representativos, so that se vea cómo evoluciona la energía.
35. As an integrante del grupo, I want contar en cuántas recuperaciones sincrónicas la energía subió entre dos iteraciones, so that se pueda contrastar con la demostración del apunte, que supone un cambio por vez.
36. As an integrante del grupo, I want verificar que en modo asincrónico la energía nunca sube, so that se confirme la demostración del apunte.
37. As an integrante del grupo, I want ver lado a lado el QR original, el patrón de consulta y la salida de la red, so that el informe muestre visualmente lo que pasa.
38. As an integrante del grupo, I want pasar el original, la consulta y la salida de la red por un lector de QR real, so that se responda si Hopfield ayuda a escanear el QR o si el lector, con su propia corrección de errores, ya lo resolvía.
39. As an integrante del grupo, I want que el lector reciba los QR ampliados y con margen blanco, so that un fallo de lectura se deba al contenido y no al formato.
40. As an integrante del grupo, I want que todo el análisis se rehaga con un solo comando, so that cualquier integrante pueda regenerar las figuras.
41. As an integrante del grupo, I want que lo generado no se suba a git, so that se mantenga la convención del ejercicio 1.
42. As an integrante del grupo, I want un informe del ejercicio 2 con la ortogonalidad, el análisis de ruido, las iteraciones, la energía y las limitaciones, so that tengamos el material para la entrega y la presentación.
43. As an integrante del grupo, I want que el informe use los términos del glosario, so that todos escribamos y expongamos con el mismo lenguaje.
44. As a docente que corrige, I want que las fórmulas del informe coincidan con las del apunte, so that pueda verificar que se implementó el modelo visto en clase.
45. As a docente que corrige, I want que las limitaciones estén respaldadas con números medidos, so that la conclusión no sea una opinión.
46. As a docente que corrige, I want entender por qué la red no recupera los QR, so that vea que el grupo comprende el crosstalk y los estados espurios.
47. As an integrante del grupo, I want tests sobre la interfaz pública de la red con patrones sintéticos chicos, so that pueda confiar en el modelo antes de usarlo con los QR.
48. As an integrante del grupo, I want que los tests que necesitan las imágenes reales se salteen si faltan los datos, so that no se sumen a los fallos que ya existen por datos ausentes.

## Implementation Decisions

- **Modelo de Hopfield (módulo nuevo, profundo).** Se le pasan los patrones almacenados (una matriz de p × N con valores ±1) y calcula la matriz de pesos una sola vez con la regla de Hebb del apunte: W = (1/N)·Σ ξξᵀ, simétrica y con diagonal cero. No se aprende nada más: W queda fija. Rechaza patrones que no sean ±1 o que no tengan dimensiones coherentes.
- **Interfaz de recuperación.** Una operación recibe un patrón de consulta, el modo de actualización, el tope (por defecto 100) y una semilla, y devuelve un resultado de recuperación con: estado final, cantidad de iteraciones, energía del estado inicial y de cada iteración, motivo de parada (estado estable / ciclo / tope) y la secuencia de estados sólo si se pide (para los ejemplos visuales).
- **Clasificación de resultados.** Separada de la recuperación: dado el estado final, el motivo de parada, los patrones almacenados y el índice del patrón de origen, devuelve la categoría (recuperación correcta / otro patrón almacenado / patrón inverso / estado espurio) y la distancia de Hamming al original. Sólo un estado estable puede ser estado espurio (definición del apunte); una recuperación que termina por ciclo o tope se informa con su motivo de parada en lugar de una categoría de estado estable.
- **Modo sincrónico.** S(t+1) = sign(W·S(t)), con campo cero → conserva el estado. Se detecta ciclo cuando el estado nuevo ya fue visitado en esa recuperación.
- **Modo asincrónico.** Una iteración es una pasada por las N neuronas en un orden aleatorio nuevo generado con un generador local sembrado por recuperación; cada neurona usa los estados ya actualizados en esa pasada. Mismo tope y mismo criterio de estado estable.
- **Energía.** H = −½·sᵀWs, equivalente a la fórmula del apunte. Se registra después de cada iteración en ambos modos.
- **Separación de responsabilidades.** Igual que Kohonen y Oja, el modelo recibe matrices NumPy y no sabe de archivos, imágenes ni gráficos.
- **Carga de QR (módulo chico).** Lee una imagen, la pasa a escala de grises, umbraliza (oscuro → +1, claro → −1), verifica que sea de 37×37 y la aplana en orden fila por fila. También hace la conversión inversa (vector → imagen) para las figuras y para el lector de QR. Las imágenes se usan tal cual; no hay preprocesamiento porque ya vienen en 37×37 y binarias, sin margen blanco.
- **Análisis de los QR (script).** En orden: (1) ortogonalidad: matriz de solapamientos, módulos compartidos, crosstalk por QR, comparación con 0,15·N; (2) estabilidad de cada QR sin ruido y comprobación del estado mezcla; (3) barrido de ruido: 5 QR × 11 niveles × 20 semillas = 1100 patrones de consulta, generados una vez y pasados por los dos modos; (4) tabla de resultados y figuras; (5) experimento secundario con el lector de QR.
- **Lector de QR.** OpenCV como experimento secundario dentro del análisis, no como métrica principal. Los patrones se amplían y se les agrega un margen blanco antes de leerlos. Se registra si cada uno (original, consulta, salida) se decodifica y si el texto coincide con el del original.
- **Fuera del runner.** El análisis no se integra a la CLI de entrenamiento ni a los barridos reanudables con checkpoints: Hopfield no se entrena por épocas y las 1100 recuperaciones tardan segundos.
- **Reproducibilidad.** Cada consulta y cada orden asincrónico se derivan de semillas fijas registradas en la tabla de resultados.
- **Salidas.** Tabla, figuras y JSON regenerables van a la carpeta de reportes del ejercicio 2 y se agregan al `.gitignore`, como en el ejercicio 1. El informe en Markdown sí se versiona.
- **Dependencias.** Se agregan Pillow (lectura de imágenes, ya instalado pero no declarado) y opencv-python (lector de QR) a los requerimientos.
- **Regla de aprendizaje.** Sólo Hebb (ADR 0001). No se ofrece ninguna otra regla en la interfaz.

## Testing Decisions

- **Qué es un buen test acá.** Se testea sólo el comportamiento observable desde la interfaz pública del modelo: qué estado devuelve la recuperación, con cuántas iteraciones, por qué se detuvo, cómo evoluciona la energía y en qué categoría cae. No se testea cómo se calculan internamente los campos ni el orden de las operaciones.
- **Punto de testeo principal: la interfaz pública del modelo de Hopfield**, con patrones sintéticos chicos:
  - patrones ortogonales almacenados son estados estables, y una consulta con un error termina en recuperación correcta;
  - el ejemplo del apunte (ξ₁ = (1,1,−1,−1), ξ₂ = (−1,−1,1,1), consulta (1,−1,−1,−1)) llega a ξ₁ en una iteración y se queda;
  - la matriz de pesos es simétrica, de diagonal cero, y coincide con la del ejemplo del apunte;
  - con tres patrones, el estado mezcla es estable y se clasifica como estado espurio; el patrón inverso se clasifica como patrón inverso;
  - un caso armado en el que el modo sincrónico oscila entre dos estados se detiene como ciclo, y el mismo caso en modo asincrónico termina en un estado estable;
  - en modo asincrónico la energía nunca sube entre iteraciones;
  - una neurona con campo cero conserva su estado;
  - con el tope en 1 y una consulta que no se estabiliza en una iteración, el motivo de parada es tope;
  - la misma semilla da exactamente el mismo resultado en modo asincrónico.
- **Punto de testeo secundario: la carga de QR**, contra las cinco imágenes reales: cada una da 1369 valores ±1, la conversión ida y vuelta es exacta y la relación negro = +1 se respeta. Se saltea si faltan los datos locales.
- **Sin tests:** el script de análisis y el experimento con el lector de QR, igual que los análisis del ejercicio 1.
- **Prior art:** la suite de TP4 usa `unittest` con datos sintéticos y aserciones de NumPy; en particular, el test de reproducibilidad de Kohonen (dos `fit` con la misma configuración dan los mismos pesos) es el modelo para el test de semilla.

## Out of Scope

- Las letras de 5×5 de la sección 2.2: es una recomendación de la consigna y el grupo decidió no hacerla.
- Cualquier regla de aprendizaje distinta de Hebb, incluida la pseudoinversa (ADR 0001).
- Modo de actualización asincrónico (quitado después de implementarlo; ver la nota del principio).
- Daño estructurado (esquina rota, rayas, manchas): sólo ruido aleatorio por módulo, que es lo que pide la consigna.
- Quitar los módulos compartidos o almacenar subconjuntos de QR para mejorar la recuperación. *(Agregado después, a pedido de Ivo, como experimento aparte: `analysis/hopfield_qr_variante.py` y `reports/ejercicio2/variante/README.md`. El análisis principal no cambia.)*
- Integración con la CLI, el runner de experimentos y los checkpoints del ejercicio 1.
- Tests del análisis y del lector de QR.
- Publicar en GitHub: la rama queda local hasta que Ivo la revise.

## Further Notes

- Números medidos en sesión el 2026-10-20 sobre las cinco imágenes (no versionado, a reproducir en el análisis): solapamientos entre 0,576 y 0,860; 955 de 1369 módulos idénticos en los cinco QR; con Hebb y modo sincrónico, los cinco QR sin ruido caen en una iteración a sign(ξ₁ + … + ξ₅); con Hebb entran dos QR cualquiera y nunca tres. Se espera, por lo tanto, que casi todo el barrido de ruido termine en ese estado mezcla.
- Los estados estables son los mismos en los dos modos de actualización (dependen sólo de sign(W·s) = s); lo que puede cambiar es el camino, la presencia de ciclos y el atractor al que se llega.
- El apunte tiene un error de transcripción en el desarrollo del crosstalk: el segundo término es ξᵢ^ν, no (1/N)·ξᵢ^ν. El informe debe escribirlo bien.
- La demostración de que la energía decrece en el apunte supone que cambia una neurona por vez; con actualización sincrónica no está garantizado. El conteo de subidas de energía es la evidencia.
- Trabajo en la rama local `tp4-ej2-hopfield`, commits a nombre de Ivo.
