# Presentación TP5

Abrir `index.html` en el navegador. Funciona sin conexión: reveal.js está en `vendor/` y los gráficos y audios en `assets/`.

- `S`: notas del orador (las explicaciones están ahí, no en las diapositivas).
- `F`: pantalla completa. `Esc`: vista general.
- Los botones ▶ reproducen un sonido por vez; cambiar de diapositiva lo corta.

La versión vigente tiene **70 diapositivas**, incluidos los agregados de audio y figuras. Reemplaza a las propuestas alternativas; `index.html` es el único archivo de presentación. Los 92 botones de audio conservan sus recursos.

La tabla siguiente conserva la numeración de la base de 58 diapositivas, anterior a los agregados.

## Consigna → diapositivas de la base original

| Consigna | Diapositivas |
|---|---|
| 1.a.1 Arquitectura con latente 2D | 5, 12–14 |
| 1.a.2 Arquitecturas y optimización estudiadas | 6–11 |
| 1.a.3 Las letras en el espacio latente | 16 |
| 1.a.4 Generar una letra nueva | 18–20 |
| 1.b.1 Arquitectura para denoising y por qué | 22–23, 25 |
| 1.b.2 Distintos niveles de ruido | 24, 26–27 |
| 2.a Conjunto de datos nuevo (NSynth) | 30–32 |
| 2.b Esquema variacional, el problema de la representación y activaciones | 34–41 |
| 2.c Generar muestras nuevas | 43–45 |
| Extras: tema de techno y notas omitidas | 46–47 |
| Apéndice | 50–58 |

## Regenerar

Desde `TP5/`:

```
python -m scripts.presentacion          # gráficos, espectrogramas y audios en reports/presentacion/
python presentation/sync_assets.py      # copia a assets/ lo que usa index.html
```

`python -m scripts.presentacion font`, `audio`, `redes`, `activaciones` u `omitidas` regenera una sola parte. `activaciones` necesita antes las corridas de `configs/ej2_{ae,vae}_activations_he.json` (leaky ReLU) y `configs/ej2_{ae,vae}_activations_xavier.json` (tanh y sigmoide), que se entrenan con `python main.py <config>`. Los gráficos de `reports/ej1_pruebas/` salen de `python -m scripts.ej1_figuras` sobre los experimentos `ej1a_stability`, `ej1a_small_failures`, `ej1b_denoising_architectures` y `ej1b_noise_types`; `omitidas` lee la evidencia guardada en `reports/ej2_notas_omitidas/puntos_medios.csv`. Al reentrenar en otra máquina los números pueden moverse un poco (redondeo de punto flotante), sin cambiar las conclusiones. `FUENTES.json` indica de qué archivo sale cada asset.
