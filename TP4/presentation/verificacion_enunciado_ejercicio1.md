# Ejercicio 1: contraste con el enunciado

Revisión del 25/10/2026, `main` en `ca3229f`, con los experimentos locales previos preservados. Fuente: [PDF del TP](../docs/SIATP42026.pdf), páginas 1–4; los requisitos específicos están en las páginas 3 y 4. Se revisaron también código, informes y las diapositivas actuales. Esta verificación corresponde al ejercicio 1, no asigna una nota ni certifica toda la entrega del TP.

## Resultado

**Hay evidencia para cubrir los cuatro apartados explícitos del ejercicio 1.** No se encontró que falte implementar uno de los tres métodos pedidos ni responder una de las preguntas orientadoras principales. Hay simplificaciones en la presentación y una diferencia entre el JSON de ejemplo y el análisis final que deben aclararse para una defensa y reproducción consistentes.

El PDF pide seleccionar variables, realizar EDA/PCA y justificar si se reduce la dimensión. No obliga a alimentar Kohonen con componentes de PCA. El pedido de agrupar considerando todas las variables se interpreta junto con la selección justificada del punto anterior: el modelo final considera conjuntamente las 27 seleccionadas, no solamente followers o PC1. Conviene dejar explícita esa justificación de las cuatro exclusiones.

## Cobertura de requisitos

| Requisito del PDF | Evidencia encontrada | Presentación | Estado |
| --- | --- | --- | --- |
| Analizar el catálogo mediante EDA | `analysis/eda_pca.py`, `analysis/eda.py`, informe de ejercicio 1 | 4–10 | Cubierto |
| Seleccionar y justificar variables | Exclusión de cuatro derivadas, 27 variables, comparación de representaciones y escalado | 4, 7–8 y apéndice 68 | Cubierto; mantener justificación explícita |
| Realizar PCA e interpretar componentes | SVD, varianza explicada, biplots, interpretación de PC1/PC2/PC3 | 9, 11 y apéndice 62 | Cubierto |
| Justificar reducción o ausencia de reducción | Comparación con PCA 2/3/6/10, decisión de conservar 27 variables | 11 y apéndice 72 | Cubierto; PCA 10 es aproximación de regiones, no identidad exacta |
| Implementar Kohonen | `src/tp4/kohonen.py`, prototipos y actualización de vecindad | 13–19 | Cubierto |
| Variar hiperparámetros y justificar performance | Informes de geometría, representación y tasa/radio; vecindad uniforme frente a gaussiana (26/10); QE, TE, ocupación y ARI | 13–14 y apéndice 64–68 | Cubierto por la evidencia informada por el grupo y por la comparación de vecindades del 26/10 |
| Analizar grilla y distribución de creadores | Ocupación, mapas y perfiles de neuronas/regiones | 13–21 | Cubierto |
| Analizar matriz U y distancia alta entre vecinas | Media de distancias de prototipos vecinos, interpretación de fronteras | 15–16 | Cubierto |
| Comparar regiones con PC1/PC2 | Proyección de las regiones sobre biplot y planos de componentes | 18–22 y apéndice 71 | Cubierto |
| Implementar Oja sobre los mismos datos | Oja recibe las mismas 27 variables estandarizadas | 24–25 | Cubierto |
| Interpretar índice y discutir máximo universal | Índice visibilidad/profundidad, signo arbitrario, filtro comercial necesario | 24–25 | Cubierto |
| Comparar Oja con proyección PC1 | Alineación, scores, ranking y varianza explicada | 24 y su informe | Cubierto |
| Tomar decisiones de campañas según evidencia | Regiones objetivo, tema y costo por alcance argentino | 27–32 | Cubierto con supuestos explícitos |
| Fundamentar fútbol: alcance, descubrimiento o profundidad | Posición en PC1, audiencia argentina, comparación de criterios | 28–29 | Cubierto como decisión razonada, no conversión validada |
| Comparar con estrategia basada en followers | Costos, perfiles y discusión de ventajas/desventajas | 31–33 | Cubierto |
| Discutir nuevas campañas, creadores y variables | Proyección con referencia guardada, revisión/reentrenamiento según cambio | 34 | Cubierto |
| Discutir actualización mensual | Cambios de medias/desvíos, componentes, regiones y referencias | Notas de 34 e informes | Cubierto; no reducirlo a aplicar un ranking fijo |

El PDF aclara que el foco es aprendizaje no supervisado e interpretación, permite supuestos sobre features y no exige un análisis exhaustivo de marketing. La ausencia de ventas/adopciones impide afirmar mejoras reales de conversiones, pero no impide responder el ejercicio con indicadores y supuestos declarados.

## Flujo real de implementación

```text
CSV: 50 creadores
  → seleccionar 27 variables y estandarizar
      → PCA: ejes, scores, biplots y varianza
      → Kohonen: 16 prototipos de 27 dimensiones
          → Ward contiguo: cinco regiones
          → interpretación y decisión humana de regiones objetivo
          → filtrar tema y ordenar por costo
      → Oja: vector de pesos e índice comparable con PC1
```

Las tres ramas comparten la representación seleccionada. `analysis/som_campanas.prepare` calcula scores PCA para interpretar y graficar, pero devuelve también la matriz de 27 variables. `train_maps` recibe esa matriz, no las dos columnas del biplot. La comparación alternativa con entradas PCA está en `analysis/som_pca_comparison.py`.

`src/tp4/pca.compute_pca` centra y usa `np.linalg.svd`, con componentes en filas, scores, autovalores calculados como singularidades al cuadrado divididas por n−1, y fracciones de varianza. La estandarización ocurre antes, en el preprocesamiento o la preparación del EDA. Se fijan signos para hacer reproducibles las salidas.

## Aspectos para aclarar antes de decir 100 % sin reservas

1. **Configuración de reproducción.** `configs/kohonen.json` es un ejemplo 5×5/500 épocas. El análisis final usa `analysis.som_campanas.MODEL`: 4×4, 1.000 épocas y decaimiento en las primeras 500. Ejecutar ese JSON no reproduce los gráficos finales. El comando del análisis final está documentado, pero es conveniente distinguir o alinear los puntos de entrada.
2. **PCA 10 no produce regiones exactamente idénticas.** El informe da ARI 0,91 con 27 variables, dos cambios de región y 12/12 recomendaciones iguales. La frase abreviada de la diapositiva 11 requiere este matiz.
3. **Tema es parte de la entrada.** La separación temática del SOM no usa etiquetas de grupo objetivo, pero sí porcentajes temáticos entre las 27 coordenadas.
4. **Decisiones comerciales.** `TARGET_REGIONS` es una decisión humana apoyada en perfiles, no un resultado aprendido a partir de ventas. Un menor costo estimado por alcance argentino no prueba más ventas o adopciones.
5. **Alcance experimental.** Repetir semillas sobre los mismos cincuenta creadores no equivale a validar en cincuenta creadores nuevos. Los experimentos locales de remuestreo identifican sensibilidad de arte; conviene mencionarla si se defiende estabilidad general.

No se modificaron las implementaciones, las configuraciones o las diapositivas en esta revisión. Las observaciones se documentan para distinguir cobertura del enunciado, exactitud del relato y facilidad de reproducción.

## Fuentes

- [Consigna](../docs/SIATP42026.pdf), páginas 3–4.
- [Presentación](index.html), diapositivas y notas 4–34, más apéndice.
- [PCA](../src/tp4/pca.py), [preprocesamiento](../src/tp4/preprocessing.py), [Kohonen](../src/tp4/kohonen.py), [Oja](../src/tp4/oja.py).
- [Análisis de campañas](../analysis/som_campanas.py), [análisis de Oja](../analysis/oja_indice.py).
- [Comparación PCA/Kohonen](../reports/ejercicio1/som_pca/README.md), [decisiones comerciales](../reports/ejercicio1/campanas/README.md).
- [Confirmaciones locales](../reports/ejercicio1/confirmacion_20261022/README.md), [augmentation local](../reports/ejercicio1/augmentation_20261022/README.md), [vecindad uniforme](../reports/ejercicio1/uniforme_20261026/README.md), [versus de vecindades](../reports/ejercicio1/versus_vecindades_20261026/README.md).
