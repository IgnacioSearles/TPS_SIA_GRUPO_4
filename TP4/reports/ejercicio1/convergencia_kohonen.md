# Duración de Kohonen: continuación de 500 a 1.000 épocas

Medición del 18/10/2026 (Argentina). La pausa del barrido de geometría dejó 23 corridas completas de 500 épocas: veinte 2×2 (dos topologías, dos inicializaciones y cinco semillas) y tres 3×3 rectangulares con inicialización desde muestras.

Se continuaron sus checkpoints hasta 1.000 sin repetir las primeras 500 épocas. El decaimiento original se conserva en 500 épocas y luego se mantienen aprendizaje 0,01 y radio 0,5. No es equivalente a empezar de cero con un decaimiento de 1.000 épocas. Los checkpoints originales y su ejecución pausada permanecen separados de las continuaciones; cada corrida nueva conserva procedencia y hash del checkpoint fuente.

| Grilla y topología | Inicialización | Semillas | Mejora media de QE 500→1.000 | QE a 1.000 | TE 500→1.000 |
| --- | --- | ---: | ---: | ---: | --- |
| 2×2 rectangular | Muestras | 5 | 0,499% | 3,666866 | 0,26→0,24 |
| 2×2 rectangular | Uniforme | 5 | 0,588% | 3,666854 | 0,26→0,24 |
| 2×2 hexagonal | Muestras | 5 | 0,080% | 3,663858 | 0→0 |
| 2×2 hexagonal | Uniforme | 5 | 0,084% | 3,663850 | 0→0 |
| 3×3 rectangular | Muestras | 3 | 1,092% | 3,222981 | 0,28→0,28 |

Entre las épocas 900 y 1.000, el mayor cambio absoluto de QE de una corrida fue 0,0273%. Comparando medias de ventanas 901–950 y 951–1.000, el mayor cambio absoluto fue 0,0103%. Ninguna corrida obtuvo su mínimo de QE en la última época. Las 23 cumplen la heurística de meseta del diagnóstico: cambio absoluto entre medias menor a 0,5%, amplitud de QE de la última ventana menor a 1% y cambio de TE medio de hasta 0,02.

**Decisión:** para estas configuraciones la extensión hasta 1.000 ayuda, pero ya no aparece una mejora sostenida al final; no se dispara la extensión condicional hasta 2.000. La meseta es operativa, bajo tasa y radio finales constantes; no garantiza un óptimo global ni permite concluir sobre grillas mayores. El error topográfico también debe evaluarse al elegir el mapa: menor QE no implica simultáneamente mejor organización.

Los artefactos locales están en:

- Origen: `results/kohonen_suite/01_geometry/20261018T234439-d4fxy0cr`.
- Continuación: `results/kohonen_continuation/1000/20261018T235225-teu_0wor`.
- Comparación: `results/kohonen_continuation/comparison_1000/`, con `paired_500_1000.csv`, `paired_summary.csv`, `runs.csv`, `summary.csv`, `snapshot.json` y `curves.png`.

Para actualizar el diagnóstico: `python3 -m analysis.kohonen_convergence --root results/kohonen_continuation/1000 --epochs 1000 --output results/kohonen_continuation/comparison_1000`.
