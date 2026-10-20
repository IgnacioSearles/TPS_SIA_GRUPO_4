# Hopfield sólo con la regla de Hebb, aunque los 5 QR no quedan almacenados

Con la regla de Hebb del apunte, ninguno de los 5 QR es un estado estable, ni siquiera sin ruido: los cinco caen en un paso al estado mezcla sign(ξ₁ + … + ξ₅). Esto pasa porque los QR se solapan entre 0,58 y 0,86 y comparten 955 de sus 1369 módulos, y no por la cantidad de patrones (5 ≪ 0,15·N ≈ 205). Decidimos igualmente usar sólo Hebb, que es el modelo visto en clase, y presentar el colapso como la limitación central del caso de uso.

## Considered Options

- **Regla de la pseudoinversa**: deja los 5 QR estables, pero no está en el apunte.
- **Quitar los módulos que comparten los cinco**: tampoco alcanza; los QR siguen sin ser estables.
- **Guardar menos QR**: con Hebb entran 2 QR cualquiera y nunca 3, pero la consigna pide almacenar los cinco.

## Consequences

El análisis de ruido de los QR no muestra una tasa de recuperación correcta: muestra adónde y en cuántas iteraciones cae la red. Los números salen de una medición en sesión (2026-10-20) sobre `data/hopfield/*.png`.
