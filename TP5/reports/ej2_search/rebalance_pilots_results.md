# Resultado de los pilotos de beta

Se completaron dos ramas pareadas de 20 épocas, semilla 7, desde exactamente
los mismos pesos padres. Se verificaron sus hashes, configuración idéntica
salvo beta y coordenadas de evaluación iguales. Adam 0,0001, batch 64,
1.280 actualizaciones por rama. MSE y KL mantienen sus implementaciones originales.

| Beta | KL media de entrenamiento | MSE de reconstrucción de entrenamiento | IoU de validación | Limpias/64 | Limpias/256 |
|---|---:|---:|---:|---:|---:|
| 0.02 | 28.923 | 0.00015420 | 0.93401 | 62 | 235 |
| 0.1 | 22.320 | 0.00017348 | 0.93353 | 62 | 235 |

El aumento de beta reduce claramente la KL por entrada, pero no mejora limpieza
en estos conjuntos. La IoU es casi igual y algo menor en la rama beta=0,1.
La inspección de las dos galerías completas (128 imágenes) conserva formas
ambiguas muy similares a las del modelo padre. No hay una mejora que justifique
promover esta afinación a tres semillas. Una sola semilla y una afinación corta
no permiten concluir que una KL mayor nunca ayude.

El resultado es compatible con el [compromiso entre información y ajuste al prior](interpretacion_regularizacion.md).
Reducir la KL media no demuestra por sí solo mejor correspondencia del agregado
con el prior ni mejor calidad. Se mantienen los criterios previos y la prueba
final cerrada. Los samplers ajustados son diagnósticos diferentes del generador
estándar; no se usa su limpieza para aprobar el experimento original.

La siguiente prioridad sería estudiar la distribución de códigos y su cobertura,
con revisión de formas y cercanía al entrenamiento. Esta tanda no respalda seguir
sumando capas o aumentar beta de esta manera como solución inmediata.

La evidencia de las dos ramas se conserva en `rebalance_pilots_evidence/`.
