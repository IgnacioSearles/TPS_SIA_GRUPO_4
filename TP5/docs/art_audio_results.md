# Revisión: arte geométrico y sonidos

Se completaron seis modelos nuevos de arte (dos datasets por tres semillas).
Se reutilizaron los tres modelos existentes de timbres, sin reentrenarlos. Las
146 pruebas pasaron, incluyendo separación de datos, RGB, GIF, medias posteriores,
reutilización de exportaciones y conservación de pesos al evaluar.

## Qué comparten

Son modelos independientes que comparten implementación y configuración del VAE:
latente 4, capas [128,32], error cuadrático, beta 0,5 con calentamiento de 50 épocas,
Adam 0,001, 200 épocas y particiones 1.024/256. Aprenden de RGB o espectrogramas;
no reciben parámetros geométricos ni armónicos. No comparten pesos ni un latente
conjunto. La pregunta común es reconstruir ejemplos reservados, generar muestras
plausibles y recorrer variaciones dentro de una familia aprendida.

## Resultados de arte

- Figuras individuales: MSE reservado 0.006297 ± 0.000096; tres modelos de 200 épocas, 226–230 s de entrenamiento por modelo.
- Composiciones de dos figuras: MSE reservado 0.009151 ± 0.000185; tres modelos de 200 épocas, 237–248 s de entrenamiento por modelo.

Los desvíos son entre las tres semillas. Estos valores no se comparan con el MSE
mel del audio. Tampoco establecen un ranking de calidad entre datasets de distinta
complejidad. El fondo fijo ocupa gran parte de la imagen: un MSE pequeño puede
coexistir con errores relevantes en las figuras.

Se inspeccionaron las seis galerías completas: 192 generaciones de figuras
individuales y 192 de composiciones. Las reconstrucciones y recorridos prefijados
permiten revisar las mismas referencias sin elegir las mejores a posteriori.

- **Individuales:** hay variaciones reconocibles de color y formas, pero también
  generaciones con ruido de color y bordes difusos. En las reconstrucciones
  inspeccionadas, algunos triángulos y rectángulos se vuelven redondeados.
- **Composiciones:** se reconoce la disposición general de dos objetos y cambios
  de color; muchos objetos quedan como manchas redondeadas, se fusionan o pierden
  sus contornos geométricos. También aparecen muestras ruidosas. No se demuestra
  conservación confiable de todas las formas y superposiciones.
- **Recorridos:** muestran cambios continuos, particularmente de color; esa
  continuidad no garantiza que los estados intermedios sean formas válidas ni
  que los atributos estén separados en coordenadas latentes independientes.

El resultado visual es parcial: permite una demostración del procedimiento y de
sus límites, pero no corresponde presentarlo como un generador de pinturas
detalladas o de geometría siempre precisa. El experimento anterior en gris se
conserva como antecedente; cambió también su pérdida y beta, por lo que no es
un control que aísle exclusivamente el efecto de agregar color.

## Relación con sonidos

En ambos casos generamos una representación de alta dimensión mediante un mismo
procedimiento de aprendizaje. En audio se mantiene la altura nominal y varían
armónicos/envolventes; en arte varían atributos visuales. No afirmamos una
equivalencia entre color y timbre ni descubrimiento autónomo de escalas.

La revisión anterior del audio registró 48/48 generaciones dentro de 50 cents de
220 Hz. Eso es un diagnóstico de afinación, no una garantía de calidad tímbrica.
Las pérdidas de armónicos superiores observadas ya afectan al códec. No se hizo
escucha subjetiva en esta revisión; quedan WAV originales, pasados por el códec,
generaciones y recorridos para escuchar antes de cerrar las conclusiones musicales.

## Qué mostrar

1. Los datasets originales y sus variaciones, explicando que los parámetros no son entradas del VAE.
2. El mismo esquema encoder–distribución latente–decoder para dos modelos independientes.
3. Reconstrucciones reservadas, incluyendo errores de forma y pérdidas del códec.
4. Galerías completas de generaciones y vecinos de entrenamiento.
5. Tres recorridos prefijados por semilla: GIF para arte, WAV para audio.
6. Las composiciones como aumento de dificultad y límite del presupuesto actual.

No se ajustaron hiperparámetros después de observar estas referencias. Un estudio
posterior de capacidad debería definir validación separada y conservar esta prueba
como evaluación, sin convertir los ejemplos más favorables en un criterio de selección.

## Material y reproducción

- [Protocolo, comandos y guion](protocolo.md).
- [Métricas por corrida](runs.csv) y [resumen por aplicación](summary_by_application.csv).
- Comparaciones compactas de índices 0,8,...,56: [semilla 7](generaciones_semilla7.png),
  [17](generaciones_semilla17.png), [27](generaciones_semilla27.png).
- Galerías completas, reconstrucciones, GIF y WAV dentro de `art_single/`,
  `art_composition/` y `timbres/`, con las tres semillas en cada aplicación.
- `manifest.json` y `source.json` registran procedencia y hashes de los modelos fuente.
