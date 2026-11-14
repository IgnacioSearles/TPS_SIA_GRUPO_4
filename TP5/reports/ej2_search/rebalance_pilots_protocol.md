# Pilotos de regularización después de aprender el decoder

Se declaran antes de observar sus generaciones. Semilla 7, dos afinaciones de
20 épocas desde el mismo `cnn4k_lowbeta_conv2_50`, batch 64, Adam 0,0001,
sin calentamiento nuevo. Comparan beta=0,02 contra beta=0,1, manteniendo todos
los demás parámetros y las pérdidas existentes. No se cambia MSE ni KL.
Cada rama hereda pesos y reinicia Adam/RNG del mismo modo. Checkpoints cada cinco.

Motivo: el diagnóstico del posterior indica que el decoder conserva mejor las
figuras cerca de códigos de entrenamiento que al muestrear desde el prior.
Una KL mayor podría reducir ese desacople, a costa de reconstrucción o nitidez.
La pregunta se prueba generando desde N(0,I), no aceptando otro sampler como
si fuese el prior entrenado. No se supone éxito ni una causalidad exclusiva.

Se evalúan referencias y generaciones de validación ya definidas, más auditoría
de 256 casos. No se abre el conjunto final. Es un piloto de una semilla: cualquier
resultado favorable requiere réplicas y revisión visual, sin cambiar umbrales.
Las presentaciones y assets existentes permanecen intactos.
