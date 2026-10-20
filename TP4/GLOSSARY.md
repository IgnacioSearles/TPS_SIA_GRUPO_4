# TP4 — Memoria asociativa (Hopfield)

Lenguaje del ejercicio 2: recuperar códigos QR deteriorados con una red de Hopfield. Las definiciones siguen el apunte `SIA_Unidad9_Aprendizaje_No_Supervisado.md`.

## Language

### Patrones

**Patrón almacenado**:
Uno de los vectores de ±1 que la red guarda en sus pesos: uno de los 5 QR originales.
_Avoid_: patrón de entrenamiento, memoria

**Patrón de consulta**:
El vector que se le presenta a la red para que lo asocie con un patrón almacenado; en este TP, un patrón almacenado con ruido.
_Avoid_: estímulo, input, query

**Módulo**:
Cada celda negra o blanca de un QR; un QR de 37×37 tiene 1369 módulos, uno por neurona. Negro = +1, blanco = −1.
_Avoid_: píxel (para los QR)

**Nivel de ruido**:
La probabilidad p con la que se invierte, de forma independiente, cada elemento de un patrón almacenado para construir un patrón de consulta.
_Avoid_: deterioro, daño (cuando se habla del ruido simulado)

### Relación entre patrones

**Solapamiento**:
El producto interno entre dos patrones dividido por N; vale 1 si son iguales, −1 si uno es el inverso del otro y 0 si son ortogonales.
_Avoid_: similitud, correlación

**Crosstalk**:
La parte del campo de una neurona que aportan los patrones almacenados distintos del consultado; es cero cuando los patrones son ortogonales.
_Avoid_: interferencia, ruido cruzado

**Capacidad**:
La cantidad máxima de patrones que la red puede almacenar y recuperar; según el apunte, p ≤ 0,15·N.

### Dinámica

**Modo de actualización**:
La regla con la que se recalculan los estados: sincrónico (todas las neuronas a la vez, como en el apunte) o asincrónico (una neurona por vez, en orden aleatorio). Es una propiedad de la recuperación, no de la red: la misma red puede usar cualquiera de los dos.
_Avoid_: red sincrónica, red asincrónica

**Iteración**:
Un recálculo completo de las N neuronas: una aplicación de sign(W·S) en modo sincrónico, o una pasada por todas las neuronas en modo asincrónico.
_Avoid_: paso, época

**Energía**:
H = −½·Σᵢⱼ wᵢⱼ Sᵢ Sⱼ, evaluada sobre el estado de la red después de cada iteración.

**Tope**:
La cantidad máxima de iteraciones permitidas antes de cortar una recuperación que no llegó a un estado estable.

**Ciclo**:
Una recuperación en la que la red vuelve a un estado que ya había visitado sin haber llegado a un estado estable.
_Avoid_: oscilación (salvo para el caso de período 2)

### Resultados de una recuperación

**Estado estable**:
Un estado que no cambia al recalcularlo; el estado al que evoluciona la red se llama atractor.
_Avoid_: punto fijo, convergencia (como sustantivo del estado)

**Recuperación correcta**:
La red termina exactamente (Hamming 0) en el patrón almacenado del que salió el patrón de consulta.

**Patrón inverso**:
El opuesto −ξ de un patrón almacenado; tiene la misma energía que ξ y también es estable, pero se reporta aparte.

**Estado espurio**:
Un estado estable que no es ningún patrón almacenado (definición del apunte); en este TP se excluye además el patrón inverso.
_Avoid_: falso atractor, error

**Estado mezcla**:
Un estado espurio formado por el signo de la suma de varios patrones almacenados; por ejemplo sign(ξ₁ + … + ξ₅).
