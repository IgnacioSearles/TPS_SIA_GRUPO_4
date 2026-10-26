# Ensayo diapositiva por diapositiva

Revisión del 25/10/2026, presentación después del pull a `ca3229f`. Hay 66 diapositivas de relato, una separadora de apéndice y 28 figuras de respaldo: 95 en total. Numeración actualizada el 26/10 con las diapositivas de dos redes, pseudoinversa y vecindad uniforme. Las preguntas son para comprobar comprensión; las respuestas se desarrollan en la [guía de estudio](guia_estudio.md).

Los enlaces abren la diapositiva correspondiente en `index.html`. La tecla S muestra las notas originales. Ninguna diapositiva fue editada para esta revisión.

## Relato principal

| Nº | Diapositiva | Leer en la guía | Pregunta que tenés que poder responder |
| --- | --- | --- | --- |
| 1 | [Elegir creadores y recordar QR](index.html#/0) | §0 | ¿Qué problemas distintos resolvemos y qué salida esperamos en cada uno? |
| 2 | [Campañas de marketing](index.html#/1) | §0 | ¿Qué quieren los cuatro clientes y qué resultados comerciales no están disponibles? |
| 3 | [Exploración y PCA](index.html#/2) | §0 | ¿Qué aporta cada etapa antes de elegir creadores? |
| 4 | [31 variables, pero 4 son cuentas de otras](index.html#/3) | §1 | ¿Qué cuatro variables se excluyen y por qué una derivada puede usarse después para costo? |
| 5 | [Cada creador tiene un tema principal](index.html#/4) | §1 | ¿Cómo se determina el tema principal y por qué un especialista no es un error? |
| 6 | [Perros es el tema más distinto](index.html#/5) | §1 | ¿Qué tendencias de perros observamos y por qué no bastan para elegir nombres? |
| 7 | [Sin escalar, los seguidores lo deciden todo](index.html#/6) | §1 | ¿Cómo se calcula z-score y qué redundancias no corrige? |
| 8 | [Casi todo el catálogo es un solo eje](index.html#/7) | §1 | ¿Por qué muchas variables correlacionadas pueden dominar la distancia? |
| 9 | [Qué significan PC1 y PC2](index.html#/8) | §2 | ¿Qué es una componente, qué significan scores y flechas y qué pasa si invertimos el signo? |
| 10 | [Los seguidores predicen el precio, no dónde está la audiencia](index.html#/9) | §2 | ¿Qué relación se observó y por qué no demuestra una causa de ventas? |
| 11 | [10 componentes guardan el 90 %, pero no reducimos](index.html#/10) | §2 | ¿Por qué PCA 10 es válido aunque se mantengan 27 variables? ¿Las regiones son idénticas? |
| 12 | [Kohonen](index.html#/11) | §3–4 | ¿Qué aprende un SOM y qué dimensiones tienen sus pesos? |
| 13 | [4×4: con 50 creadores, un mapa más grande se vacía](index.html#/12) | §3–4 | ¿Por qué no elegir el mapa de menor QE? ¿Qué ARI y configuración se comparan? |
| 14 | [Gaussiana o uniforme: mismo error, distinto mapa](index.html#/13) | §3–4 | ¿Qué se dejó afuera de la calibración? ¿Por qué el mismo radio no significa lo mismo en las dos vecindades y por qué mantenemos gaussiana? |
| 15 | [Las fronteras de la matriz U son cambios de perfil](index.html#/14) | §4 | ¿Cómo calcula Grid.u_matrix cada celda y qué significa un valor alto? |
| 16 | [Las regiones: juntamos neuronas vecinas parecidas](index.html#/15) | §4 | ¿Cómo une Ward las neuronas y qué significa el tamaño de una región en su fórmula? |
| 17 | [Con 5 regiones aparecen los temas, sin regiones diminutas](index.html#/16) | §4 | ¿Qué compromiso sostiene cinco regiones? ¿Más pureza siempre significa mejor mapa? |
| 18 | [En diagonal, el mapa va de visibilidad a profundidad](index.html#/17) | §4 | ¿Qué está fijo en la grilla y qué valor se pinta en cada plano? |
| 19 | [Y cada tema ocupa su propio rincón](index.html#/18) | §4 | ¿Qué datos temáticos entraron al SOM? ¿Por qué no es un descubrimiento sin información temática? |
| 20 | [Las regiones recorren PC1 de punta a punta](index.html#/19) | §4 | ¿Cómo puede el SOM separar perfiles que el plano PC1-PC2 superpone? |
| 21 | [Sin pedírselo, el mapa separa los temas](index.html#/20) | §4 | ¿Cómo se calcula pureza y qué controla el contraste con regiones al azar? |
| 22 | [Lo que el mapa no separa: la audiencia argentina](index.html#/21) | §4 | ¿Qué mide eta cuadrado y por qué necesitamos ordenar después por costo? |
| 23 | [Índice de Oja](index.html#/22) | §5 | ¿Qué cambia entre un vector Oja y los 16 prototipos del SOM? |
| 24 | [La regla de Oja encuentra PC1](index.html#/23) | §5 | ¿Podés ejecutar una actualización de Oja y explicar cómo se mide la alineación con PC1? |
| 25 | [¿Elegimos al de mayor índice? No](index.html#/24) | §5 | ¿Por qué un índice alto no significa mejor creador? ¿Qué cambia al invertir el signo? |
| 26 | [Decisiones para las campañas](index.html#/25) | §6 | ¿Qué decisiones aparecen después de los modelos y quién las fija? |
| 27 | [Cómo elegimos](index.html#/26) | §6 | ¿Dónde está TARGET_REGIONS y cómo se calcula CPM_AR? |
| 28 | [Masivo y descubrimiento están en la misma punta de PC1](index.html#/27) | §6 | ¿En qué catálogo alcance y descubrimiento apuntan al mismo lado? ¿Es una ley general? |
| 29 | [Fútbol: ¿alcance masivo, descubrimiento o profundidad?](index.html#/28) | §6 | ¿Por qué elegimos visibilidad local para fútbol y qué alternativa razonable queda? |
| 30 | [Lo que quiere cada cliente elige la región](index.html#/29) | §6 | ¿Qué perfil justifica cada región objetivo y cuál es el caso menos robusto? |
| 31 | [Llegar a cada argentino cuesta entre 15 % y 47 % menos](index.html#/30) | §6 | ¿Qué costo se compara y por qué no equivale a ventas ni a alcance único deduplicado? |
| 32 | [Refugios: tres creadores chicos le ganan a tres famosos](index.html#/31) | §6 | ¿Qué muestran USD 339 contra USD 6.993 y qué no prueban sobre adopciones? |
| 33 | [¿Seguidores o mapa?](index.html#/32) | §6 | ¿Qué agrega SOM más costo frente al ranking simple y qué mantenimiento exige? |
| 34 | [¿Qué pasa cuando algo cambia?](index.html#/33) | §6 | ¿Cuándo se proyecta sin entrenar y cuándo se revisa el preprocesamiento y el modelo? |
| 35 | [En resumen](index.html#/34) | §6 | ¿Podés resumir el ejercicio sin presentar los supuestos comerciales como resultados medidos? |
| 36 | [Memoria asociativa](index.html#/35) | §7, §12 | ¿Qué es memoria asociativa y cómo pasamos de 37x37 módulos a un vector? |
| 37 | [Qué le pedimos a la red](index.html#/36) | §7, §12 | ¿Qué diferencia hay entre almacenar, recuperar exactamente y decodificar? |
| 38 | [Le pasamos cualquier QR y devuelve siempre lo mismo](index.html#/37) | §7, §12 | ¿Cuántas consultas hubo, a dónde fueron y por qué falla incluso con ruido cero? |
| 39 | [La red baja de energía, pero hacia la mezcla](index.html#/38) | §7, §12 | ¿Cómo puede bajar energía y empeorar la identidad? ¿Qué supone la garantía de monotonía? |
| 40 | [La salida ni siquiera es un QR que se pueda leer](index.html#/39) | §7, §12 | ¿Qué hace el lector y por qué lectura y exactitud son medidas distintas? |
| 41 | [Problema 1: los cinco QR no son ortogonales](index.html#/40) | §7–8 | ¿Qué significa casi ortogonal para patrones bipolares? |
| 42 | [¿Está mal la red? Con patrones al azar anda perfecto](index.html#/41) | §7–8 | ¿Qué se mantiene fijo en el control aleatorio y qué diagnóstico apoya? |
| 43 | [Tampoco es cantidad](index.html#/42) | §7–8 | ¿Qué supuestos tiene 0,15N y por qué no es una garantía para QR correlacionados? |
| 44 | [Los cinco QR son casi el mismo QR](index.html#/43) | §7–8 | ¿Cómo se relacionan solapamiento y porcentaje de coincidencias? ¿Qué incluye el 70 % común? |
| 45 | [El crosstalk le gana a la señal en 71 a 225 neuronas por QR](index.html#/44) | §7–8 | ¿Qué se colorea en rojo? ¿Por qué una media grande de crosstalk no basta para explicar errores? |
| 46 | [Con más de 25 % compartido, los otros patrones le ganan al propio](index.html#/45) | §7–8 | ¿Bajo qué aproximación sale 1-4q y por qué 25 % no es universal? |
| 47 | [El parecido viene de cómo se arma un QR](index.html#/46) | §7–8 | ¿Qué comparten estas URL y por qué no debemos generalizar a todos los QR de un sitio? |
| 48 | [Qué hicimos con el problema 1](index.html#/47) | §11 | ¿Qué causa se busca cambiar con cada variante? |
| 49 | [Cortar la imagen en zonas no alcanza](index.html#/48) | §11 | ¿Por qué piezas recuperadas por separado pueden producir un QR completo inválido? |
| 50 | [Sacar lo común: entran de a tres, pero los cinco juntos siguen en cero](index.html#/49) | §11 | ¿Cómo funcionan strip y restore? ¿Cuántos originales quedan estables con Hebb y cinco QR? |
| 51 | [Problema 2: sin lo común, menú y vinos siguen pareciéndose](index.html#/50) | §8 | ¿Qué problema queda después de retirar las coordenadas comunes? |
| 52 | [Sacar lo común separa a casi todos los pares, menos a menú y vinos](index.html#/51) | §8 | ¿Por qué cambian los solapamientos y por qué el par menú-vinos sigue destacándose? |
| 53 | [Las ternas que fallan son las que tienen a menú y vinos](index.html#/52) | §8 | ¿Qué comparación entre ternas apoya el papel de ese par y cuál es su alcance? |
| 54 | [Por qué ese par: en los datos difieren en 6 bits](index.html#/53) | §8 | ¿Cómo se obtienen los 6 bits de datos y los 90 de corrección? ¿Qué se reconstruyó exactamente? |
| 55 | [Qué hicimos con el problema 2](index.html#/54) | §11 | ¿Qué alternativas cambian el reparto y cuáles cambian los pesos? |
| 56 | [Separar el par: menú en una red, vinos en otra](index.html#/55) | §11 | ¿Qué identidades almacena cada red y a qué barrido pertenece el 100 % hasta 35 %? |
| 57 | [Pero elegir la red es el problema 2 otra vez](index.html#/56) | §11 | ¿Qué optimiza el reparto? ¿Por qué desempatar comparando con los originales es Hamming y no la red? |
| 58 | [Sin mirar los originales: elegir la red por energía](index.html#/57) | §11 | ¿Por qué la energía con Hebb sigue midiendo parecido con los QR guardados? ¿Qué agrega frente a Hamming? |
| 59 | [Desaprender la mezcla: una sola red guarda los cinco](index.html#/58) | §10, §12 | ¿Qué estado se desaprende? ¿Cómo funciona anti-Hebb? ¿Por qué 50 intentos no son 50 updates? |
| 60 | [Con desaprendizaje también llega en 1 o 2 iteraciones, pero al QR](index.html#/59) | §10, §12 | ¿Qué cuenta una iteración y por qué 2.100 evaluaciones no son consultas independientes? |
| 61 | [Pseudoinversa: otra forma de calcular los pesos](index.html#/60) | §9 | ¿Qué descuenta (Ξ Ξᵀ)⁻¹ y por qué cada QR guardado queda estable? |
| 62 | [Ninguna red le gana a comparar con los originales](index.html#/61) | §9, §12 | ¿Qué métodos usaron las mismas 700 consultas y qué niveles entran en la media? ¿Cómo funciona pinv? |
| 63 | [Límites y conclusiones](index.html#/62) | §12, §14 | ¿Cuáles son límites del algoritmo y cuáles del diseño de evaluación? |
| 64 | [La esquina rota casi no le llega a la red](index.html#/63) | §12, §14 | ¿Cuántos módulos cubre, altera y ve la red? ¿Qué arregla la plantilla? |
| 65 | [Límites del caso de uso](index.html#/64) | §12, §14 | ¿Qué alternativas solo se proponen? ¿Los porcentajes de corrección QR son porcentajes de píxeles? |
| 66 | [Los cinco QR se parecen. Sin lo común, menú y vinos siguen pareciéndose.](index.html#/65) | §12, §14 | ¿Cómo reformularías la conclusión para no afirmar que sacar lo común basta con cinco QR? |

## Apéndice

| Nº | Diapositiva | Leer en la guía | Pregunta que tenés que poder responder |
| --- | --- | --- | --- |
| 67 | [Apéndice](index.html#/66) | §0 | ¿Qué gráfico de respaldo elegirías para cada pregunta del docente? |
| 68 | [Escalas originales y estandarizadas](index.html#/67) | §1 | ¿Cómo verificamos que el escalado cambió el peso de magnitudes, sin eliminar correlaciones? |
| 69 | [Biplot PC1–PC2 con todos los creadores](index.html#/68) | §2 | ¿Qué distancias e interpretaciones del biplot son aproximadas? |
| 70 | [PC3: publicidad y fútbol](index.html#/69) | §2 | ¿Qué información de publicidad y fútbol no se ve en PC1-PC2? |
| 71 | [Correlaciones por bloques, con valores](index.html#/70) | §1–2 | ¿Cómo leer bloques positivos y oposiciones sin atribuir causalidad? |
| 72 | [Barrido de geometría del mapa](index.html#/71) | §4 | ¿Qué factores se variaron y qué efecto del radio puede confundirse con forma? |
| 73 | [Barrido de tasa de aprendizaje y radio](index.html#/72) | §3–4 | ¿Cómo afecta el radio a QE y TE y cómo decaen tasa y radio? |
| 74 | [Vecindad uniforme con las mismas semillas y datos](index.html#/73) | §3–4 | ¿Por qué uniforme 2 → 0,5 baja QE de entrenamiento y empeora en excluidos? |
| 75 | [Qué radio eligió cada vecindad al calibrar](index.html#/74) | §3–4 | ¿En cuántas particiones ganó un radio constante y qué dice eso del radio menor a 1 en uniforme? |
| 76 | [Representaciones de entrada para Kohonen](index.html#/75) | §2, §4 | ¿Por qué QE no se compara directamente entre representaciones de distinta dimensión? |
| 77 | [Planos de componentes del mapa](index.html#/76) | §4 | ¿Cómo se pasa del peso estandarizado a unidades originales en un plano? |
| 78 | [Qué variables separan las regiones](index.html#/77) | §4 | ¿Cómo se calcula eta cuadrado y cuál fue la referencia al azar? |
| 79 | [Regiones en PC1–PC3](index.html#/78) | §2, §4 | ¿Qué separa PC3 que no muestra el plano principal? |
| 80 | [Kohonen sobre componentes principales](index.html#/79) | §2 | ¿Qué cambió al entrenar con 2, 3, 6 y 10 componentes? |
| 81 | [Candidatos de cada cliente](index.html#/80) | §6 | ¿Por qué un candidato barato de otra región puede no ser del tipo comercial elegido? |
| 82 | [Elegidos frente al top 3 por seguidores](index.html#/81) | §6 | ¿Qué parte del ahorro viene de mirar público argentino y no solo seguidores? |
| 83 | [Regiones y elegidos en el mapa](index.html#/82) | §6 | ¿Qué se hace si un elegido no está disponible y qué no garantiza cercanía en el mapa? |
| 84 | [Un QR con ruido, paso a paso](index.html#/83) | §7 | ¿Podés seguir consulta, estados intermedios y salida sin confundir cercanía con estabilidad? |
| 85 | [Iteraciones hasta converger](index.html#/84) | §7 | ¿Por qué pocas iteraciones no implican buena recuperación? |
| 86 | [Solapamientos entre los QR](index.html#/85) | §7 | ¿Cómo se calcula cada entrada de la matriz de solapamientos? |
| 87 | [Interferencia por módulo](index.html#/86) | §7 | ¿Cómo se obtiene el campo exacto con diagonal cero? |
| 88 | [Capacidad medida: patrones al azar contra QR](index.html#/87) | §8 | ¿Con qué criterio se midió capacidad y por qué difiere entre QR y patrones aleatorios? |
| 89 | [Sin compartidos: ruido, con la reposición sin red como referencia](index.html#/88) | §11 | ¿Qué mejora aporta solo reponer comunes sin Hopfield? |
| 90 | [Sin compartidos: todas las ternas](index.html#/89) | §8, §11 | ¿Qué ternas contienen al par más parecido y cómo se compararon? |
| 91 | [Una red por zona: barrido de ruido y lector](index.html#/90) | §11 | ¿Cómo difieren exactitud de módulos y tasa de lectura por tamaño de zona? |
| 92 | [Cuántas redes y cómo repartir los QR](index.html#/91) | §11 | ¿Qué costo conceptual tiene aumentar cantidad de redes hasta una por recuerdo? |
| 93 | [Cómo se elige la red](index.html#/92) | §11 | ¿Cuándo el selector elige un recuerdo reconocido y cuándo una salida espuria? |
| 94 | [Todos los métodos con una esquina rota](index.html#/93) | §12 | ¿Las alternativas de este gráfico comparten consultas con la confirmación nueva? |
| 95 | [Desaprendizaje: ruido, esquinas y manchas](index.html#/94) | §10, §12 | ¿Dónde se rompe el 100 % del desaprendizaje y cuánto del daño ve realmente la red? |
