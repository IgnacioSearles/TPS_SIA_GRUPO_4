# Revisión visual del candidato conv2

Se inspeccionaron las 192 salidas de validación, sin filtros ni exclusión de
fallos. Las etiquetas del asistente son subjetivas, conservadoras y no ciegas;
no constituyen una evaluación humana independiente. Los mosaicos mantienen los
índices originales y muestran por separado el resultado de limpieza automática.

El criterio previo exigía cinco casos limpios inequívocos de cada familia por
semilla. Se reconocieron solamente 3 círculos en la semilla 7, 5 en la 17 y 4
en la 27. Por lo tanto **no pasa la revisión formal** y no se congela ni se abre
la prueba final. No se cambian los mínimos para declarar éxito.

Hay figuras identificables con bordes imperfectos y formas híbridas que no se
fuerzan a una categoría. Por ejemplo, en la semilla 27 las muestras 13 y 61
pasan limpieza automática, pero tienen geometría irregular. Esto muestra que
la limpieza y la conservación de familias son preguntas diferentes.

`cases.csv` conserva los 192 casos. `review.json` guarda etiquetas, índices,
conteos limpios y hashes de las matrices generadas. Las categorías visuales
describen la salida, no la supuesta intención del modelo.
