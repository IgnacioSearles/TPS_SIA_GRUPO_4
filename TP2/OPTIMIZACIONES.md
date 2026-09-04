# Posibles optimizaciones para el TP2

Este documento resume optimizaciones posibles para mejorar la calidad visual,
la velocidad de convergencia y el rendimiento del algoritmo que aproxima imagenes
con triangulos semitransparentes.

La idea no es listar variantes que ya estan implementadas, sino proponer cambios
con potencial real para una siguiente iteracion del proyecto. Por eso no se
incluyen como propuestas principales cosas que el codigo ya cubre: metricas como
`mse`, `regional`, `ssim`, `blur`, `multiscale`, `histogram`, `edge`, `gradient`,
`chamfer`, `saliency`, combinaciones de fitness, seleccion por ranking,
Boltzmann, torneos, ruleta, universal, ni schedules basicos de mutacion.

## Resumen de prioridad

| Prioridad | Optimizacion | Impacto esperado | Complejidad | Area |
| --- | --- | --- | --- | --- |
| 1 | Mutacion guiada por error de imagen | Muy alto | Media | Mutacion / fitness |
| 2 | Color estimado desde el target | Muy alto | Media | Mutacion / representacion |
| 3 | Busqueda local por triangulo | Muy alto | Media-alta | Optimizacion |
| 4 | Modelo de islas | Alto | Media-alta | Arquitectura GA |
| 5 | Evaluacion parcial o incremental | Alto | Alta | Rendimiento |
| 6 | Evolucion progresiva por resolucion | Alto | Media | Runner / configuracion |
| 7 | Paralelizar evaluaciones | Medio-alto | Media | Rendimiento |
| 8 | Mutaciones geometricas compuestas | Medio-alto | Media | Mutacion |
| 9 | Orden de genes como parte de la mutacion | Medio | Baja-media | Mutacion |
| 10 | Primitivas mixtas | Medio-alto | Alta | Representacion |
| 11 | Color perceptual Lab / DeltaE | Medio | Media | Fitness |
| 12 | Reportes de diagnostico extra | Medio | Baja | Observabilidad |

## 1. Mutacion guiada por error de imagen

### Problema actual

Las mutaciones actuales cambian genes de forma bastante aleatoria: pueden tocar
color, posicion, forma, orientacion o reemplazar un triangulo entero. Eso cumple
con el enfoque de algoritmo genetico general, pero desperdicia muchas
evaluaciones.

Si una imagen tiene una zona claramente mal aproximada, por ejemplo una parte de
la camiseta o del escudo, una mutacion puramente aleatoria puede modificar un
triangulo en una zona que ya estaba razonablemente bien. Esa mutacion tiene poca
probabilidad de mejorar el resultado.

### Idea

Calcular un mapa de error entre la imagen renderizada y el target:

```text
error_pixel = diferencia entre target y render
```

Despues usar ese mapa para orientar la mutacion:

- elegir con mas probabilidad triangulos que cubren zonas con mucho error;
- mover triangulos hacia regiones con error alto;
- reemplazar triangulos ubicandolos en zonas mal aproximadas;
- inicializar nuevos triangulos cerca de bordes o zonas visualmente importantes;
- elegir el tamano de la mutacion segun el error local.

El algoritmo deja de mutar completamente a ciegas. Sigue siendo estocastico, pero
usa informacion del problema.

### Ejemplo conceptual

En vez de:

```text
elegir triangulo random
mutar posicion random
```

se podria hacer:

```text
renderizar mejor individuo
calcular mapa de error
elegir una zona proporcional al error
elegir o crear un triangulo cerca de esa zona
mutarlo
```

### Por que tiene mucho potencial

En este problema, el fitness no es una caja negra total: sabemos que el objetivo
es una imagen y podemos mirar donde esta el error. Esa informacion es muy valiosa.

Un paper sobre el problema "Evolution of Mona Lisa" muestra justamente que
incorporar operadores de mutacion centrados en la imagen puede acelerar la
convergencia. La idea general es que las mutaciones no solo cambien parametros
del cromosoma, sino que tengan relacion con el contenido visual del target.

### Riesgos

Si se guia demasiado la mutacion, el algoritmo puede perder diversidad. Conviene
mezclar:

```text
70% mutacion guiada por error
30% mutacion aleatoria tradicional
```

No deberia reemplazar por completo la mutacion random.

## 2. Color estimado desde el target

### Problema actual

El color del triangulo se busca por mutacion: se cambia RGB y alpha con ruido.
Eso obliga al algoritmo a resolver dos problemas a la vez:

```text
donde poner el triangulo
que color darle
```

El espacio de busqueda se vuelve enorme. Un triangulo puede estar muy bien
ubicado, pero tener un color malo y por eso ser descartado.

### Idea

Cuando se crea o muta un triangulo, estimar su color usando los pixeles del
target en la zona que cubre.

Version simple:

```text
color_triangulo = promedio RGB del target dentro del triangulo
```

Version mejor:

```text
color_triangulo = color que mejor reduce la diferencia entre canvas actual y target
```

La segunda version es mas potente porque no mira solo el target, sino tambien lo
que ya esta dibujado. Si el canvas actual ya tiene algo de rojo en esa zona, el
nuevo triangulo no necesariamente necesita ser rojo puro.

### Por que mejora

Reduce mucho el azar. La geometria todavia se busca evolutivamente, pero el color
puede salir de una estimacion razonable. Esto evita gastar generaciones
descubriendo colores obvios.

Herramientas como Primitive y Geometrize usan esta idea: para una figura dada,
calculan el color que mas conviene sobre los pixeles afectados.

### Variante facil para implementar primero

Para cada triangulo nuevo o reemplazado:

1. calcular sus vertices;
2. rasterizar una mascara del triangulo;
3. tomar los pixeles del target dentro de la mascara;
4. asignar el promedio RGB;
5. elegir alpha dentro de un rango razonable, por ejemplo `0.15` a `0.65`.

No es optimo matematico, pero ya seria bastante mejor que RGB completamente
random.

## 3. Busqueda local por triangulo

### Problema actual

El algoritmo genetico prueba poblaciones completas. Eso sirve para exploracion
global, pero puede ser lento para ajustes finos. Muchas veces una solucion esta
cerca de mejorar si se mueve un triangulo unos pocos pixeles, si se rota un poco
o si se cambia levemente su alpha.

Un GA puro puede tardar muchas generaciones en hacer ese ajuste chico.

### Idea

Agregar una etapa de busqueda local despues de generar un hijo. Por ejemplo:

```text
hijo = crossover + mutacion
repetir N veces:
    probar pequena perturbacion en un triangulo
    si mejora el fitness, conservarla
    si empeora, descartarla
```

Esto convierte el algoritmo en un enfoque hibrido o memetico:

```text
GA -> explora combinaciones grandes
busqueda local -> mejora detalles de cada candidato
```

### Aplicacion concreta

Se podria aplicar solo sobre:

- el mejor individuo de cada generacion;
- los mejores `k` hijos;
- un porcentaje bajo de la poblacion;
- un triangulo elegido por error local.

Eso evita multiplicar demasiado el costo.

### Por que tiene mucho potencial

El problema de aproximar imagenes con figuras geometricas tiene muchisimos
parametros continuos: coordenadas, tamano, angulos, rotacion, RGB y alpha.
La busqueda local suele funcionar bien cuando una pequena perturbacion puede
mejorar gradualmente la solucion.

Primitive reporta un enfoque parecido: generar una forma, mutarla muchas veces y
quedarse con cambios que reducen RMSE. En este problema concreto, ese estilo de
hill climbing es muy competitivo.

## 4. Modelo de islas

### Problema actual

En una poblacion unica, todos los individuos compiten entre si. Si aparece una
familia de soluciones con fitness razonablemente alto, puede dominar rapido. Eso
reduce diversidad y puede dejar al algoritmo atrapado en una aproximacion mala.

Este problema se nota mucho en imagenes dificiles: la poblacion puede aprender
una mancha general de colores, pero no salir de ahi.

### Idea

Dividir la poblacion en varias subpoblaciones independientes, llamadas islas:

```text
isla 1 -> poblacion A
isla 2 -> poblacion B
isla 3 -> poblacion C
isla 4 -> poblacion D
```

Cada isla evoluciona por separado durante varias generaciones. Cada cierto
intervalo migran algunos individuos entre islas:

```text
cada 100 generaciones:
    mejor 2 de isla 1 migran a isla 2
    mejor 2 de isla 2 migran a isla 3
    mejor 2 de isla 3 migran a isla 4
    mejor 2 de isla 4 migran a isla 1
```

Al final, el resultado es simplemente el mejor individuo global entre todas las
islas.

### Como converge a una sola imagen

Cada individuo de cada isla sigue representando una imagen completa. Las islas no
son pedazos de la imagen. No hay una isla para la camiseta, otra para el fondo y
otra para la cara.

Todas intentan resolver el mismo target completo:

```text
individuo = imagen candidata completa
```

La migracion permite que una solucion buena descubierta en una isla sea
aprovechada por otra. Despues la cruza puede combinar genes de distintos
origenes.

### Variantes utiles

Misma configuracion en todas las islas:

```text
4 islas
misma fitness
misma mutacion
distintas semillas
```

Configuraciones distintas:

```text
isla 1 -> mas peso en MSE
isla 2 -> mas peso en bordes
isla 3 -> mas peso en saliency
isla 4 -> mutacion mas fuerte
```

La segunda version es mas interesante, pero tambien mas dificil de analizar.

### Parametros importantes

- cantidad de islas;
- tamano de poblacion por isla;
- frecuencia de migracion;
- cantidad de migrantes;
- criterio para elegir migrantes: mejores, aleatorios o mezcla;
- criterio de reemplazo en destino: peores o aleatorios;
- topologia: anillo, todos contra todos, grilla.

Una version razonable para empezar:

```text
4 islas
20 individuos por isla
migracion cada 100 generaciones
migran los mejores 2
reemplazan los peores 2
topologia en anillo
```

### Por que mejora

Mantiene diversidad sin depender solamente de subir la mutacion. Subir mucho la
mutacion puede destruir soluciones buenas; las islas permiten explorar en
paralelo sin mezclar todo permanentemente.

Tambien es naturalmente paralelizable: cada isla puede correr en un proceso
distinto.

## 5. Evaluacion parcial o incremental

### Problema actual

Evaluar un individuo implica renderizar todos sus triangulos y comparar toda la
imagen contra el target. Eso se repite miles o millones de veces.

Pero muchas mutaciones cambian solo una parte del individuo. Si se modifica un
triangulo, no siempre seria necesario recalcular todo desde cero.

### Idea

Guardar informacion de la imagen renderizada y recalcular solo la parte afectada.

Version conceptual:

```text
fitness_actual = error total actual
mutar triangulo
calcular zona afectada por ese triangulo
actualizar error solo en esa zona
fitness_nuevo = fitness_actual ajustado
```

### Dificultad especial

En este proyecto los triangulos son semitransparentes y el orden importa. Si se
muta un triangulo temprano del genoma, cambia como se ven todos los triangulos
que estan encima. Por eso no alcanza con redibujar solo ese triangulo aislado.

Una implementacion seria:

- guardar renders parciales por prefijo del genoma;
- al mutar el gen `i`, partir desde el canvas hasta `i - 1`;
- redibujar desde `i` hasta el final;
- comparar solo el bounding box afectado cuando sea posible.

### Por que igual vale la pena

Aunque sea compleja, esta optimizacion puede bajar mucho el tiempo de ejecucion.
Geometrize documenta una funcion de diferencia parcial para recalcular el error
solo en las zonas donde una forma cambia. Primitive tambien menciona partial
image difference como una optimizacion importante.

## 6. Evolucion progresiva por resolucion

### Problema actual

Correr directamente con una imagen de 256 o 300 px y muchos triangulos hace que
el espacio de busqueda sea grande desde el inicio.

Al principio no hace falta optimizar detalles finos. Primero conviene aprender
la composicion general:

- grandes masas de color;
- siluetas;
- ubicacion general del objeto;
- contraste principal.

### Idea

Ejecutar la evolucion en etapas:

```text
etapa 1 -> max_size 64, mutacion fuerte, pocos triangulos o misma cantidad
etapa 2 -> max_size 128, mutacion media
etapa 3 -> max_size 256/300, mutacion fina
```

El mejor individuo de una etapa se escala y se usa como poblacion inicial o como
semilla para la siguiente.

### Diferencia con fitness multiscale

El proyecto ya tiene una metrica multiescala, pero esto es otra cosa. La metrica
multiescala evalua varias resoluciones dentro del fitness. La evolucion
progresiva cambia la resolucion real del problema a lo largo del tiempo.

### Por que mejora

Reduce costo al principio y evita que el algoritmo se distraiga con detalles
microscopicos cuando todavia no resolvio la estructura grande de la imagen.

## 7. Paralelizar evaluaciones

### Problema actual

La evaluacion de individuos es independiente:

```text
fitness(individuo A)
fitness(individuo B)
fitness(individuo C)
```

No hay dependencia entre esos calculos dentro de la misma generacion.

### Idea

Evaluar hijos y poblacion en paralelo usando procesos.

Esto no necesariamente mejora la cantidad de generaciones necesarias, pero baja
el tiempo de pared si hay CPU disponible.

### Aplicacion

El punto natural es el metodo que evalua individuos en el orquestador. En vez de
usar un loop secuencial, se podria usar un pool de workers.

### Cuidado

Hay que medir bien porque:

- serializar individuos entre procesos tiene costo;
- PIL y numpy pueden consumir memoria;
- si la poblacion es chica, el overhead puede comerse la mejora;
- en macOS crear procesos puede ser relativamente caro.

Conviene activarlo por config:

```json
{
  "runtime": { "workers": 4 }
}
```

## 8. Mutaciones geometricas compuestas

### Problema actual

El mutador actual puede cambiar posicion, color, forma u orientacion. Pero muchas
transformaciones utiles no son solo tocar un parametro: son transformaciones
geometricas coherentes.

### Ideas de mutaciones nuevas

Trasladar el triangulo completo:

```text
center_x += dx
center_y += dy
```

Escalar manteniendo forma:

```text
size *= factor
```

Rotar alrededor de su centro:

```text
rotation += delta
```

Estirar de forma controlada:

```text
modificar angulos sin destruir el triangulo
```

Jitter de vertices:

```text
mover directamente vertices individuales
```

### Por que mejora

Estas mutaciones respetan mas la estructura visual. Si un triangulo ya esta
cerca de una zona buena, trasladarlo o escalarlo suavemente tiene mas sentido que
cambiar parametros independientes con ruido uniforme.

El paper de mutaciones image-centric para Mona Lisa menciona operadores como
translation, scaling y rotation aplicados a poligonos.

## 9. Mutacion del orden de genes

### Problema actual

Los triangulos son semitransparentes y se dibujan en orden. Por lo tanto, el
orden del genoma afecta la imagen final.

Dos individuos con los mismos triangulos pero en distinto orden pueden verse
distintos.

### Idea

Agregar mutaciones que cambien orden:

- swap de dos triangulos;
- mover un triangulo a otra posicion del genoma;
- invertir un segmento corto;
- mandar un triangulo muy transparente al fondo o al frente segun convenga.

### Por que mejora

Actualmente la cruza puede cambiar combinaciones de genes, pero si no hay una
presion explicita sobre el orden, algunas soluciones buenas pueden quedar
bloqueadas porque los triangulos correctos estan dibujados en mala capa.

Esta mejora es relativamente barata de implementar.

## 10. Primitivas mixtas

### Problema actual

Representar todo solo con triangulos es posible, pero no siempre eficiente.
Algunas zonas de una imagen se representan mejor con otras figuras:

- circulos o elipses para manchas suaves;
- rectangulos rotados para bordes rectos;
- poligonos de mas vertices para areas grandes;
- lineas o curvas para contornos finos.

### Idea

Permitir que un gen represente una primitiva de varios tipos:

```text
TriangleGene
EllipseGene
RectangleGene
PolygonGene
```

O tener un campo:

```text
shape_type = "triangle" | "ellipse" | "rectangle"
```

### Por que mejora

Con la misma cantidad de genes, una poblacion podria representar mejor ciertos
targets. Primitive usa multiples primitivas justamente porque diferentes formas
capturan distintos tipos de estructura visual.

### Riesgo

Es una mejora grande de arquitectura. Cambia codec, render, mutadores,
inicializador, documentacion y tests. No seria la primera optimizacion a
implementar si el objetivo es mejorar rapido.

## 11. Color perceptual Lab / DeltaE

### Problema actual

MSE RGB mide diferencias numericas por canal. Pero el ojo humano no percibe
igual todos los cambios RGB.

Dos errores con el mismo MSE pueden verse muy distintos. Por ejemplo, ciertos
cambios en tonos azules o grises pueden percibirse mas o menos que lo que sugiere
RGB.

### Idea

Convertir target y render a un espacio perceptual como CIELAB y medir diferencia
de color con DeltaE, idealmente CIEDE2000.

### Por que podria ayudar

CIEDE2000 fue disenado para aproximar mejor diferencias de color percibidas por
humanos que una simple distancia en RGB. Puede servir como parte de un combo:

```json
{
  "fitness": {
    "metric": "combo",
    "combo": {
      "mse": 0.30,
      "edge": 0.25,
      "saliency": 0.25,
      "deltae": 0.20
    }
  }
}
```

### Cuidado

No lo usaria como unica metrica al principio. Puede mejorar color perceptual, pero
no necesariamente estructura espacial. Conviene combinarlo con MSE, bordes o
saliency.

## 12. Reportes de diagnostico extra

### Problema actual

La fitness normalizada puede ser enganosa. Un valor como `0.94` parece excelente,
pero visualmente puede ser pobre si la normalizacion comprime mucho las
diferencias o si el fondo domina.

### Idea

Agregar al progreso y al `summary.json` metricas crudas de diagnostico:

- MSE crudo;
- RMSE;
- error promedio por canal;
- error en zonas opacas vs transparentes;
- fitness principal;
- fitness secundaria de control;
- diversidad de poblacion;
- mejor, promedio y peor fitness por generacion.

### Por que mejora

No hace que el algoritmo genere mejores imagenes directamente, pero ayuda a
entender que esta pasando. Para comparar configuraciones, esto es clave.

Ejemplo:

```text
fitness normalizada: 0.94
RMSE RGB: 61.8
error opaco: alto
error fondo: bajo
```

Eso permite detectar casos donde el algoritmo parece bueno numericamente pero
esta fallando en la parte importante de la imagen.

## Propuesta de roadmap

Una secuencia razonable de implementacion seria:

### Paso 1: mejoras de bajo riesgo

- reportar RMSE/MSE crudo;
- agregar mutacion de orden de genes;
- agregar color inicial desde promedio del target para triangulos nuevos;
- crear una config recomendada para imagenes dificiles.

### Paso 2: mejoras de alto impacto

- mutacion guiada por mapa de error;
- reemplazo de triangulos en zonas con error alto;
- busqueda local liviana sobre el mejor individuo o mejores hijos;
- evolucion progresiva por resolucion.

### Paso 3: mejoras de arquitectura

- modelo de islas;
- evaluacion paralela;
- scoring parcial o incremental;
- primitivas mixtas.

## Recomendacion concreta para el grupo

Si hubiera que elegir una sola linea de trabajo, conviene empezar por:

```text
mutacion guiada por error + color estimado desde el target
```

Es la mejora con mejor relacion entre impacto y complejidad para este codigo. No
cambia todo el motor genetico, no obliga a reescribir el render y aprovecha
informacion que el problema ya tiene disponible.

Despues, la segunda mejora mas atractiva seria:

```text
modelo de islas
```

Porque ataca la convergencia prematura y ademas permite correr variantes en
paralelo. Para imagenes dificiles puede ser mas valioso que seguir ajustando una
unica seleccion dentro de una unica poblacion.

## Referencias

- Wang, Bovik, Sheikh y Simoncelli. "Image quality assessment: From error
  visibility to structural similarity". IEEE Transactions on Image Processing,
  2004. Disponible en: <https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Wang03>
- Luo, Cui y Rigg. "The development of the CIE 2000 colour-difference formula:
  CIEDE2000". Color Research & Application, 2001. Disponible en:
  <https://doi.org/10.1002/col.1049>
- Croitoru. "Accelerating heuristic convergence on the Evolution of Mona Lisa
  problem by including image-centric mutation operators", 2022. Disponible en:
  <https://scispace.com/papers/accelerating-heuristic-convergence-on-the-evolution-of-mona-3shv6ejx>
- Primitive, de Michael Fogleman. Reproduccion de imagenes con primitivas
  geometricas. Disponible en: <https://github.com/fogleman/primitive>
- Geometrize. Documentacion de funciones como `computeColor`, `differenceFull`,
  `differencePartial` y busqueda local por shape. Disponible en:
  <https://docs.geometrize.co.uk/core_8cpp.html>
- Hansen. Introduccion a CMA-ES como referencia general sobre estrategias
  evolutivas con adaptacion de parametros continuos. Disponible en:
  <https://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaesintro.html>
