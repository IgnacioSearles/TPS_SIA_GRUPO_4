# Exploratory data analysis — fraud dataset

Fuente de definiciones: [documentación del dataset](../../datasets/fraud_dataset_documentation.pdf).

- Filas: 7500
- Columnas analizadas: 10
- Valores faltantes: 0
- Filas duplicadas: 0
- Probabilidad objetivo: media 0.4228, mínimo 0.0009, máximo 1.0000

## Columnas documentadas

| Columna | Significado / unidad | Uso |
| --- | --- | --- |
| `timestamp` | Instante de la compra, segundos Unix | Descartada |
| `amount_usd` | Monto total de la compra, USD | Entrada |
| `quantity_purchased` | Cantidad comprada de un mismo artículo | Entrada |
| `session_duration_seconds` | Duración de la sesión, segundos | Entrada |
| `days_since_last_purchase` | Días desde la última compra | Entrada |
| `account_age_days` | Antigüedad de la cuenta, días | Entrada |
| `device_screen_resolution` | Cantidad de píxeles de pantalla (ancho × alto) | Descartada |
| `time_since_last_login_s` | Segundos desde el último ingreso | Descartada |
| `items_viewed_before_purchase` | Artículos vistos antes de comprar | Entrada |
| `big_model_fraud_probability` | Probabilidad del modelo de referencia, entre 0 y 1 | Objetivo |

## Rangos observados

| Columna | Mínimo | Mediana | P99 | Máximo |
| --- | ---: | ---: | ---: | ---: |
| amount_usd | 1.000 | 63.490 | 858.047 | 2000.000 |
| quantity_purchased | 1.000 | 5.000 | 22.000 | 24.000 |
| session_duration_seconds | 5.000 | 287.000 | 571.801 | 726.800 |
| days_since_last_purchase | 0.000 | 8.590 | 67.120 | 142.320 |
| account_age_days | 1.000 | 1633.500 | 3615.000 | 3649.000 |
| items_viewed_before_purchase | 1.000 | 8.000 | 27.000 | 29.000 |
| timestamp | 1700001808.000 | 1715682901.000 | 1731199542.540 | 1731534222.000 |
| device_screen_resolution | 1006733.000 | 2073158.000 | 8302524.030 | 8310940.000 |
| time_since_last_login_s | 10.000 | 2477.700 | 15977.059 | 40160.800 |
| big_model_fraud_probability | 0.001 | 0.357 | 1.000 | 1.000 |

## Decisión de preprocesamiento

Se usan como features: amount_usd, quantity_purchased, session_duration_seconds, days_since_last_purchase, account_age_days, items_viewed_before_purchase.
Se excluyen: timestamp, device_screen_resolution, time_since_last_login_s, por su baja correlación lineal observada con la probabilidad objetivo en el análisis inicial.
`big_model_fraud_probability` es el objetivo continuo en [0, 1].
Se elige estandarización Z-score (media 0, desvío 1): centra las entradas y pone sus escalas en un rango comparable para el descenso por gradiente. También puede verse afectada por valores extremos. El scaler se ajusta solo con las filas de entrenamiento de cada fold y se reutiliza en su validación.
Se empleará K-Fold con mezcla reproducible: toda muestra valida una vez y el escalador se ajusta de nuevo en cada fold.
Se conservan histogramas y correlaciones de todas las variables originales junto a vistas de las variables elegidas.

## Correlación lineal con la probabilidad objetivo

| Variable | Correlación |
| --- | ---: |
| amount_usd | 0.557394 |
| quantity_purchased | 0.563128 |
| session_duration_seconds | -0.513803 |
| days_since_last_purchase | -0.404237 |
| account_age_days | -0.584736 |
| items_viewed_before_purchase | 0.334365 |
| timestamp | 0.001433 |
| device_screen_resolution | 0.024618 |
| time_since_last_login_s | 0.002445 |
