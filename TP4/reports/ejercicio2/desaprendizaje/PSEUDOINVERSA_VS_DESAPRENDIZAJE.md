# Pseudoinversa frente a desaprendizaje: comparación directa

Se evaluaron ambas variantes de pseudoinversa sobre las mismas 700 consultas de
confirmación del candidato de desaprendizaje: cinco QR, niveles
0/0.1/0.2/0.3/0.35/0.4/0.45, semillas de consulta 100–119 y tope de 100 iteraciones
sincrónicas. Se mantuvo fijo el desaprendizaje aleatorio seleccionado: tasa 0.1,
50 intentos, semillas de entrenamiento 0/1/2, 414 módulos activos y restauración
de 955 compartidos. Sus resultados previos se reutilizaron, sin reentrenar ni
seleccionar otra configuración después de observar esta comparación.

Se añadieron 1400 recuperaciones de pseudoinversa. Todas las alternativas de esta
tabla son redes de Hopfield; no se utiliza un selector por Hamming entre ellas.
La pseudoinversa completa usa 1369 neuronas; la reducida y el desaprendizaje usan
414 y restauran los módulos comunes. Se exige coincidencia exacta y estado estable.

| Ruido | Pseudoinversa completa | Pseudoinversa reducida | Desaprendizaje reducido |
| --- | ---: | ---: | ---: |
| 0 | 100% | 100% | 100% |
| 0.1 | 100% | 100% | 100% |
| 0.2 | 100% | 100% | 100% |
| 0.3 | 100% | 100% | 99.0% |
| 0.35 | 100% | 100% | 98.7% |
| 0.4 | 96.0% | 88.0% | 88.7% |
| 0.45 | 64.0% | 49.0% | 56.7% |
| Media de los seis niveles positivos | **93.3%** | **89.5%** | **90.5%** |

La mejor exactitud de estas tres alternativas fue la pseudoinversa completa.
En la misma representación reducida, desaprendizaje obtuvo una media ligeramente
mayor que pseudoinversa, concentrando su ventaja en el ruido alto. No corresponde
presentar este resultado como superioridad universal de una regla.

Datos individuales y comparación emparejada:
[directorio de resultados](evidencia/comparacion_pseudoinversa_20261022/).
Los porcentajes del desaprendizaje promedian las tres semillas de entrenamiento;
las 700 consultas son las mismas para cada una, no 2100 consultas independientes.
El archivo `comparacion_emparejada.json` contiene intervalos bootstrap de diferencias,
con 20000 remuestreos de semillas de consulta como bloques, semilla 20261022,
condicionados a los cinco QR y a las redes entrenadas.

## Incertidumbre de las diferencias

Bootstrap emparejado de 20000 bloques de semillas de consulta, condicionado a los
cinco QR y a las redes entrenadas:

- Desaprendizaje menos pseudoinversa completa: −2.8 puntos porcentuales, intervalo
  percentil 95% de −4.6 a −1.4.
- Desaprendizaje menos pseudoinversa reducida: +1.0 punto, intervalo de −0.2 a +2.3.
  La pequeña ventaja observada no es concluyente.

[Guía para presentar estos resultados](../../../presentation/experimentos_hopfield_nuevos.md).
