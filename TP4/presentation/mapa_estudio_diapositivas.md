# Ensayo diapositiva por diapositiva

Revisión del 25/10/2026, presentación después del pull a `ca3229f`. Hay 60 diapositivas de relato, una separadora de apéndice y 37 figuras de respaldo: 98 en total. Numeración actualizada el 27/10 con la corrida de manchas y rayones hasta 40–50 %.

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
| 14 | [Cómo se reparte la actualización entre vecinas](index.html#/13) | §3–4 | ¿Qué fórmula da cada peso? ¿Por qué con radio menor a 1 la uniforme solo mueve a la ganadora? |
| 15 | [Gaussiana o uniforme: mismo error, distinto mapa](index.html#/14) | §3–4 | ¿Qué se dejó afuera de la calibración? ¿Por qué el mismo radio no significa lo mismo en las dos vecindades y por qué mantenemos gaussiana? |
| 16 | [Las fronteras de la matriz U son cambios de perfil](index.html#/15) | §4 | ¿Cómo calcula Grid.u_matrix cada celda y qué significa un valor alto? |
| 17 | [Las regiones: juntamos neuronas vecinas parecidas](index.html#/16) | §4 | ¿Cómo une Ward las neuronas y qué significa el tamaño de una región en su fórmula? |
| 18 | [Con 5 regiones aparecen los temas, sin regiones diminutas](index.html#/17) | §4 | ¿Qué compromiso sostiene cinco regiones? ¿Más pureza siempre significa mejor mapa? |
| 19 | [En diagonal, el mapa va de visibilidad a profundidad](index.html#/18) | §4 | ¿Qué está fijo en la grilla y qué valor se pinta en cada plano? |
| 20 | [Y cada tema ocupa su propio rincón](index.html#/19) | §4 | ¿Qué datos temáticos entraron al SOM? ¿Por qué no es un descubrimiento sin información temática? |
| 21 | [Las regiones recorren PC1 de punta a punta](index.html#/20) | §4 | ¿Cómo puede el SOM separar perfiles que el plano PC1-PC2 superpone? |
| 22 | [Sin pedírselo, el mapa separa los temas](index.html#/21) | §4 | ¿Cómo se calcula pureza y qué controla el contraste con regiones al azar? |
| 23 | [Lo que el mapa no separa: la audiencia argentina](index.html#/22) | §4 | ¿Qué mide eta cuadrado y por qué necesitamos ordenar después por costo? |
| 24 | [Índice de Oja](index.html#/23) | §5 | ¿Qué cambia entre un vector Oja y los 16 prototipos del SOM? |
| 25 | [La regla de Oja encuentra PC1](index.html#/24) | §5 | ¿Podés ejecutar una actualización de Oja y explicar cómo se mide la alineación con PC1? |
| 26 | [¿Elegimos al de mayor índice? No](index.html#/25) | §5 | ¿Por qué un índice alto no significa mejor creador? ¿Qué cambia al invertir el signo? |
| 27 | [Decisiones para las campañas](index.html#/26) | §6 | ¿Qué decisiones aparecen después de los modelos y quién las fija? |
| 28 | [Cómo elegimos](index.html#/27) | §6 | ¿Dónde está TARGET_REGIONS y cómo se calcula CPM_AR? |
| 29 | [Masivo y descubrimiento están en la misma punta de PC1](index.html#/28) | §6 | ¿En qué catálogo alcance y descubrimiento apuntan al mismo lado? ¿Es una ley general? |
| 30 | [Fútbol: ¿alcance masivo, descubrimiento o profundidad?](index.html#/29) | §6 | ¿Por qué elegimos visibilidad local para fútbol y qué alternativa razonable queda? |
| 31 | [Lo que quiere cada cliente elige la región](index.html#/30) | §6 | ¿Qué perfil justifica cada región objetivo y cuál es el caso menos robusto? |
| 32 | [Primero la región, después el más barato](index.html#/31) | §6 | ¿Qué decide el mapa y qué decide el costo? ¿Por qué ningún top por seguidores de perros entra en R1? |
| 33 | [Llegar a cada argentino cuesta entre 15 % y 47 % menos](index.html#/32) | §6 | ¿Qué costo se compara y por qué no equivale a ventas ni a alcance único deduplicado? |
| 34 | [Refugios: tres creadores chicos le ganan a tres famosos](index.html#/33) | §6 | ¿Qué muestran USD 339 contra USD 6.993 y qué no prueban sobre adopciones? |
| 35 | [¿Seguidores o mapa?](index.html#/34) | §6 | ¿Qué agrega SOM más costo frente al ranking simple y qué mantenimiento exige? |
| 36 | [¿Qué pasa cuando algo cambia?](index.html#/35) | §6 | ¿Cuándo se proyecta sin entrenar y cuándo se revisa el preprocesamiento y el modelo? |
| 37 | [En resumen](index.html#/36) | §6 | ¿Podés resumir el ejercicio sin presentar los supuestos comerciales como resultados medidos? |
| 38 | [Memoria asociativa](index.html#/37) | §7, §12 | ¿Qué es memoria asociativa y cómo pasamos de 37x37 módulos a un vector? |
| 39 | [Le pasamos cualquier QR y devuelve siempre lo mismo](index.html#/38) | §7, §12 | ¿Cuántas consultas hubo, a dónde fueron y por qué falla incluso con ruido cero? |
| 40 | [La energía baja, pero hacia la mezcla](index.html#/39) | §7, §12 | ¿Por qué la mezcla tiene menos energía que cualquier QR guardado? ¿Qué supone la garantía de que la energía no sube? |
| 41 | [Problema 1: los cinco QR no son ortogonales](index.html#/40) | §7–8 | ¿Qué significa casi ortogonal para patrones bipolares? |
| 42 | [No es la red ni la cantidad](index.html#/41) | §7–8 | ¿Qué se mantiene fijo en el control al azar y qué supuestos tiene 0,15·N? |
| 43 | [Los cinco QR son casi el mismo QR](index.html#/42) | §7–8 | ¿Cómo se define el solapamiento? ¿Por qué las diferencias quedan a la izquierda del QR? |
| 44 | [Con más de 25 % compartido, los otros QR le ganan al propio](index.html#/43) | §7–8 | ¿De dónde sale 1 − 4 × compartido y por qué vale solo para el peor módulo? |
| 45 | [Sacar lo común: entran de a tres, pero los cinco juntos siguen en cero](index.html#/44) | §11 | ¿Cómo funcionan strip y restore? ¿Por qué reponer lo común no es comparar con los originales? |
| 46 | [Problema 2: sin lo común, food y wine siguen pareciéndose](index.html#/45) | §8 | ¿Qué problema queda después de retirar las coordenadas comunes? |
| 47 | [Sacar lo común separa a casi todos los pares, menos a food y wine](index.html#/46) | §8 | ¿Por qué los pares negativos no molestan y el par menú-vinos sí? ¿Qué muestran las ternas? |
| 48 | [Por qué ese par: en los datos difieren en 6 bits](index.html#/47) | §8 | ¿Cómo se llega de 96 módulos distintos a un solapamiento de 0,54? |
| 49 | [Separado el par, cada red funciona; elegir la red no](index.html#/48) | §11 | ¿Por qué las dos redes son un diagnóstico y no una solución? ¿Por qué elegir por energía sigue midiendo parecido con los QR? |
| 50 | [Soluciones: una sola red que guarde los cinco](index.html#/49) | §9–10 | ¿Qué tienen en común desaprendizaje y pseudoinversa y en qué se diferencian de las dos redes? |
| 51 | [Desaprender es Hebb al revés, solo para lo falso](index.html#/50) | §10 | ¿Qué término comparten Hebb y el desaprendizaje? ¿Por qué Hebb no puede evitar la mezcla y cuánto sube la energía de un QR al restar un estado falso? |
| 52 | [Desaprendizaje: rellenar los pozos falsos](index.html#/51) | §10 | ¿Qué estado se resta, cuándo no se toca y cuánto sube su energía? ¿Por qué 50 intentos no son 50 restas? |
| 53 | [Después de desaprender, los cinco QR son estables](index.html#/52) | §10 | ¿Qué cuenta "neuronas que lo rechazan"? ¿Qué pasó en los 50 intentos de cada semilla? |
| 54 | [Desaprendizaje: 99 % hasta 35 % de ruido, casi siempre en 1 o 2 iteraciones](index.html#/53) | §10, §12 | ¿Por qué las consultas son nuevas? ¿Qué cuenta una iteración y qué pasó con la energía? |
| 55 | [La pseudoinversa es Hebb sin el eco del parecido](index.html#/54) | §9 | ¿Por qué Hebb confunde parecido con presencia? ¿Qué descuenta G⁻¹ y por qué al mostrar food solo vota food? |
| 56 | [Pseudoinversa: medir el parecido y descontarlo](index.html#/55) | §9 | ¿Qué mide G, para qué se invierte y en qué se diferencia W = Ξᵀ G⁻¹ Ξ de Hebb? ¿Cuándo las dos reglas coinciden? |
| 57 | [Las redes quedan cerca del mejor caso posible](index.html#/56) | §9, §12 | ¿Por qué comparar con los originales es el techo con este ruido? ¿Qué diferencias son concluyentes? |
| 58 | [Con daño realista, las redes recuperan casi todo](index.html#/57) | §12 | ¿Cómo se genera cada daño? ¿Hasta qué porcentaje de módulos alterados se probó cada tipo y dónde se rompe cada método? |
| 59 | [Conclusiones](index.html#/58) | §9–12 | ¿Por qué falla Hebb, cuánto recuperan desaprendizaje y pseudoinversa, y por qué Hamming gana con ruido al azar pero no con desgaste? ¿Por qué la pseudoinversa completa supera a la de sin lo común y al desaprendizaje? |
| 60 | [Qué le recomendaríamos a la Trattoria](index.html#/59) | §12, §14 | ¿Por qué Hamming primero y la pseudoinversa completa como memoria? ¿Qué supone que la consulta llegue a 37×37 y qué alternativas solo se proponen? |

## Apéndice

| Nº | Diapositiva | Leer en la guía | Pregunta que tenés que poder responder |
| --- | --- | --- | --- |
| 61 | [Apéndice](index.html#/60) | §0 | ¿Qué gráfico de respaldo elegirías para cada pregunta del docente? |
| 62 | [Escalas originales y estandarizadas](index.html#/61) | §1 | ¿Cómo verificamos que el escalado cambió el peso de magnitudes, sin eliminar correlaciones? |
| 63 | [Biplot PC1–PC2 con todos los creadores](index.html#/62) | §2 | ¿Qué distancias e interpretaciones del biplot son aproximadas? |
| 64 | [PC3: publicidad y fútbol](index.html#/63) | §2 | ¿Qué información de publicidad y fútbol no se ve en PC1-PC2? |
| 65 | [Correlaciones por bloques, con valores](index.html#/64) | §1–2 | ¿Cómo leer bloques positivos y oposiciones sin atribuir causalidad? |
| 66 | [Barrido de geometría del mapa](index.html#/65) | §4 | ¿Qué factores se variaron y qué efecto del radio puede confundirse con forma? |
| 67 | [Barrido de tasa de aprendizaje y radio](index.html#/66) | §3–4 | ¿Cómo afecta el radio a QE y TE y cómo decaen tasa y radio? |
| 68 | [Vecindad uniforme con las mismas semillas y datos](index.html#/67) | §3–4 | ¿Por qué uniforme 2 → 0,5 baja QE de entrenamiento y empeora en excluidos? |
| 69 | [Qué radio eligió cada vecindad al calibrar](index.html#/68) | §3–4 | ¿En cuántas particiones ganó un radio constante y qué dice eso del radio menor a 1 en uniforme? |
| 70 | [Representaciones de entrada para Kohonen](index.html#/69) | §2, §4 | ¿Por qué QE no se compara directamente entre representaciones de distinta dimensión? |
| 71 | [Planos de componentes del mapa](index.html#/70) | §4 | ¿Cómo se pasa del peso estandarizado a unidades originales en un plano? |
| 72 | [Qué variables separan las regiones](index.html#/71) | §4 | ¿Cómo se calcula eta cuadrado y cuál fue la referencia al azar? |
| 73 | [Regiones en PC1–PC3](index.html#/72) | §2, §4 | ¿Qué separa PC3 que no muestra el plano principal? |
| 74 | [Kohonen sobre componentes principales](index.html#/73) | §2 | ¿Qué cambió al entrenar con 2, 3, 6 y 10 componentes? |
| 75 | [Candidatos de cada cliente](index.html#/74) | §6 | ¿Por qué un candidato barato de otra región puede no ser del tipo comercial elegido? |
| 76 | [Elegidos frente al top 3 por seguidores](index.html#/75) | §6 | ¿Qué parte del ahorro viene de mirar público argentino y no solo seguidores? |
| 77 | [Regiones y elegidos en el mapa](index.html#/76) | §6 | ¿Qué se hace si un elegido no está disponible y qué no garantiza cercanía en el mapa? |
| 78 | [Cómo funciona la red](index.html#/77) | §7 | ¿Cómo se pasa del QR a neuronas, cómo se guarda con Hebb, cómo se actualiza y cuándo para? ¿Qué mide la energía? |
| 79 | [Pseudoinversa: descontar el parecido antes de guardar](index.html#/78) | §9 | ¿Qué matriz se invierte, de qué tamaño, y por qué cada QR guardado queda estable? |
| 80 | [Pseudoinversa en chico: 6 neuronas, 3 recuerdos](index.html#/79) | §9 | ¿Por qué Hebb da vuelta la quinta neurona de A y la pseudoinversa no? |
| 81 | [Un QR con ruido, paso a paso](index.html#/80) | §7 | ¿Podés seguir consulta, estados intermedios y salida sin confundir cercanía con estabilidad? |
| 82 | [Hebb: iteraciones hasta converger](index.html#/81) | §7 | ¿Por qué pocas iteraciones no implican buena recuperación? |
| 83 | [Solapamientos entre los QR](index.html#/82) | §7 | ¿Cómo se calcula cada entrada de la matriz de solapamientos? |
| 84 | [Crosstalk: dónde el campo le gana a cada QR](index.html#/83) | §7–8 | ¿Qué se pinta en rojo y por qué está en la franja izquierda? |
| 85 | [Capacidad medida: patrones al azar contra QR](index.html#/84) | §8 | ¿Con qué criterio se midió capacidad y por qué difiere entre QR y patrones aleatorios? |
| 86 | [Sin compartidos: ruido, con la reposición sin red como referencia](index.html#/85) | §11 | ¿Qué mejora aporta solo reponer comunes sin Hopfield? |
| 87 | [Sin compartidos: todas las ternas](index.html#/86) | §8, §11 | ¿Qué ternas contienen al par más parecido y cómo se compararon? |
| 88 | [Una red por zona: QR estables en cada zona](index.html#/87) | §11 | ¿Por qué cortar en zonas no reduce el parecido? |
| 89 | [Una red por zona: barrido de ruido y lector](index.html#/88) | §11 | ¿Cómo difieren exactitud de módulos y tasa de lectura por tamaño de zona? |
| 90 | [Dos redes (diagnóstico): cuántas redes y cómo repartir](index.html#/89) | §11 | ¿Qué costo tiene llevar el reparto hasta una red por QR? |
| 91 | [Dos redes (diagnóstico): qué devuelve cada una](index.html#/90) | §11 | ¿Cuándo devuelven QR guardados las dos redes y qué regla elige? |
| 92 | [Una esquina rota cae casi toda donde los QR son iguales](index.html#/91) | §12, §14 | ¿Cuántos módulos cubre, altera y ve la red? ¿Hasta qué tamaño se recupera? |
| 93 | [Todos los métodos con una esquina rota](index.html#/92) | §12 | ¿Las alternativas de este gráfico comparten consultas con la confirmación nueva? |
| 94 | [Desaprendizaje: ruido, esquinas y manchas](index.html#/93) | §10, §12 | ¿Dónde se rompe el 100 % del desaprendizaje y cuánto del daño ve realmente la red? |
| 95 | [Daño realista: los siete tipos](index.html#/94) | §12 | ¿Cómo se genera cada daño y en cuáles el lector solo deja de leer? |
| 96 | [Daño realista: niveles extremos](index.html#/95) | §12 | ¿Dónde se rompe cada método y por qué ahí conviene saber que el daño solo borra tinta? |
| 97 | [Daño realista: manchas y rayones hasta 40–50 %](index.html#/96) | §12 | ¿Por qué se agregó esta corrida y qué cambia con manchas negras frente a manchas blancas? |
| 98 | [Daño realista: un ejemplo de cada tipo](index.html#/97) | §12 | ¿Qué parte de cada daño ve la red? |
