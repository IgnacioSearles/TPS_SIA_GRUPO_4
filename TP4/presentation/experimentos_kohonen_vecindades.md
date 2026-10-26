# Vecindad uniforme frente a gaussiana: guía para presentar

Estas notas acompañan la diapositiva 14 y el apéndice 74–75, agregados el 26/10. No cambian el modelo final ni las otras diapositivas. Informes: [tanda uniforme](../reports/ejercicio1/uniforme_20261026/README.md) y [versus calibrado](../reports/ejercicio1/versus_vecindades_20261026/README.md); [bitácora](../reports/ejercicio1/bitacora/README.md).

## Mensajes que la evidencia sostiene

### 1. Calibradas igual, cuantizan casi lo mismo (diapositiva 14)

Mostrar **QE externo 3,728 gaussiana y 3,731 uniforme**. Es el error medio sobre diez creadores que quedaron fuera de escalado, entrenamiento y elección de parámetros, en 15 particiones con tres semillas cada una. Cada familia tuvo diez candidatos: cinco radios/calendarios × tasas iniciales 0,1 y 0,3.

Notas: la diferencia cambia de signo entre particiones; uniforme gana en 8/15. No es una prueba formal de equivalencia.

### 2. Difieren en el mapa, no en el error

Mostrar **TE 24,9 % uniforme contra 34,7 % gaussiana** (uniforme gana 10/15) y **ARI regional entre semillas 0,760 gaussiana contra 0,681 uniforme**. Mantenemos gaussiana porque las campañas se deciden sobre las regiones y las suyas se repiten más.

### 3. El mismo radio no significa lo mismo (notas de la 14 y apéndice 74)

Uniforme con radio menor que 1 actualiza solo a la ganadora: con 2 → 0,5 eso pasa desde la época 251. La gaussiana de ancho 0,5 le da todavía 0,135 a una vecina inmediata. Con las mismas semillas que el modelo final, uniforme 2 → 0,5 baja QE de entrenamiento (2,65 contra 2,79), pero sube TE a 24 % y empeora en excluidos. Uniforme 2 → 1 ordena casi perfecto (TE 2,5 %) con QE 3,18.

### 4. El radio constante no es siempre malo (apéndice 75)

En la calibración, uniforme eligió radio constante en 8/15 particiones y gaussiana en 4/15. Gaussiana eligió 2 → 0,5 (nuestro modelo) en 10/15; uniforme nunca. El barrido histórico ya tenía constantes 0,5 a 8 con cinco semillas.

## Qué no afirmar

- Que gaussiana es mejor en general, o que hay equivalencia demostrada.
- Que el procedimiento confirmó una única configuración: cada partición pudo elegir otra.
- Que más orden topográfico implica mejores campañas o ventas.
- Que augmentation ayuda: en la tanda uniforme ninguna intensidad pasó todos los criterios.

## Cambios en la presentación

- Diapositiva 14 nueva, con `versus_externo.png`.
- Apéndice 74 (`uniforme_20261026/comparacion.png`) y 75 (`versus_vecindades_20261026/selecciones.png`).
- Una frase agregada al final de las notas de la 13.
- `sync_assets.py` agregó las tres figuras a `assets/` y `FUENTES.json`.

Las figuras del versus se regeneraron solo para traducir títulos y poner ticks enteros; las métricas y tablas quedaron byte a byte iguales.
