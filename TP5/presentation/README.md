# Presentación TP5

Abrir `index.html` en el navegador. Funciona sin conexión: reveal.js está en `vendor/` y los gráficos y audios en `assets/`.

- `S`: notas del orador (las explicaciones están ahí, no en las diapositivas).
- `F`: pantalla completa. `Esc`: vista general.
- Los botones ▶ reproducen un sonido por vez; cambiar de diapositiva lo corta.

## Consigna → diapositivas

| Consigna | Diapositivas |
|---|---|
| 1.a.1 Arquitectura con latente 2D | 5–6 |
| 1.a.2 Arquitecturas y optimización estudiadas | 7–13 |
| 1.a.3 Las letras en el espacio latente | 15 |
| 1.a.4 Generar una letra nueva | 17–19 |
| 1.b.1 Arquitectura para denoising y por qué | 21–22 |
| 1.b.2 Distintos niveles de ruido | 23–25 |
| 2.a Conjunto de datos nuevo (NSynth) | 28–30 |
| 2.b Esquema variacional y el problema de la representación | 32–37 |
| 2.c Generar muestras nuevas | 39–41 |
| Extras: tema de techno y notas omitidas | 42–43 |
| Apéndice | 45–54 |

## Regenerar

Desde `TP5/`:

```
python -m scripts.presentacion          # gráficos, espectrogramas y audios en reports/presentacion/
python presentation/sync_assets.py      # copia a assets/ lo que usa index.html
```

`python -m scripts.presentacion font`, `audio` o `redes` regenera una sola parte. Los gráficos de `reports/ej1_pruebas/` y `reports/ej2_notas_omitidas/` vienen de experimentos de otros integrantes y no se regeneran con estos comandos. `FUENTES.json` indica de qué archivo sale cada asset.
