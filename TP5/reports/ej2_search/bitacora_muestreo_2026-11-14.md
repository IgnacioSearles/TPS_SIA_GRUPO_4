# Bitácora adicional: distribución de muestreo y beta

Se respetó la instrucción de solo agregar en la presentación. No había cambios
propios en diapositivas, notas, estilos o assets que revertir. Se agregó la regla
en `presentation/AGENTS.md` y un plan nuevo `propuesta_solo_agregados.md`; la
propuesta previa de reemplazar una diapositiva queda descartada, sin ejecutarse.

Se probaron cuatro recorridos del latente de los tres VAE RGB existentes, con
256 casos nuevos por método y semilla: 3.072 salidas, sin entrenar. La limpieza
fue 93,5 % con prior estándar, 96,6 % con momentos ajustados, 99,9 % con mezcla
posterior y 99,9 % al reconstruir medias. Los dos últimos dependen de imágenes
de entrenamiento y quedan mucho más cerca de ellas; son diagnósticos, no una
solución equivalente ni evidencia de novedad. Ver `sampling_diagnostic_2026-11-14/`.

Se probó después una afinación pareada de 20 épocas con beta 0,02 o 0,1,
desde el mismo padre, misma tasa y mismos datos. Ambas obtuvieron 62/64 limpias
y 235/256 en auditoría; subir beta bajó la KL sin mejorar estas generaciones.
Se inspeccionaron ambas galerías y persistieron las ambigüedades geométricas.
No se promueven estos pilotos y la prueba final sigue cerrada.

Siete pruebas matemáticas del nuevo sampler pasaron. La ejecución integral
terminó en los tres modelos; se verificaron pesos y datos de origen. No se
modificaron las fórmulas protegidas. Los archivos anteriores se conservan.
