# TP3 — Presentación (45 minutos)

Presentación para los 45 minutos de exposición: 66 diapositivas de relato (14 son separadores de sección, casi sin tiempo de exposición) y un apéndice de 25 gráficos para preguntas.

Abrir index.html. Flechas: avanzar. F: pantalla completa. S: notas del orador. Funciona sin conexión. Para compartir, enviar la carpeta completa.

## Estructura

Cada ejercicio sigue las mismas secciones, con un separador numerado y una línea de recorrido que resalta la sección actual:

**x.1 Exploración de datos · x.2 Entrenamiento · x.3 Resultados · x.4 Conclusiones · x.5 Opcionales**

Los opcionales de la consigna para los ejercicios 2 y 3 son conjuntos: se presentan una sola vez, en 3.5.

| Bloque | Diapositivas | Tiempo aproximado |
|---|---|---|
| Portada | 1 | 1 min |
| Ejercicio 1 | 2–26 | 17 min |
| Ejercicio 2 | 27–43 | 14 min |
| Ejercicio 3 | 44–65 | 17 min |
| Cierre | 66 | 2 min |
| Apéndice | 67–92 | solo para preguntas |

## Dónde se responde cada pregunta de la consigna

| Pregunta | Diapositiva |
|---|---|
| E1 aprendizaje a/b/c (underfitting, saturación, elección) | 10 |
| E1 generalización a (métricas) | 15 |
| E1 generalización b (manejo de datos) | 13 |
| E1 generalización c (mejor modelo y umbral) | 18 y 20 (recomendación para CompanyX) |
| E1 opcionales: ReLU · features · calibración | 22 · 23–24 · 25–26 |
| E2 a (cómo evaluar) | 28 y 43 |
| E2 b (variantes) | 33–38, 40 (qué adoptamos) y 43 (incluye gradientes y L2) |
| E3 protocolo (partición, selección, semillas, parada temprana) | 48 |
| E3 a / b / c | 56 / 51 / 50, resumidas en 60 |
| E3 modelo elegido · consultas al test | 55 · 57 |
| E2-E3 opcionales: ruido · interpretabilidad | 62 · 63–64 |
| Extra: ensamble de redes (98,92 % en test) | 65 |

## Criterios

- Una diapositiva por idea: el gráfico y su conclusión juntos.
- El análisis detallado está en las notas del orador (S), con sus cautelas y fuentes.
- Los gráficos secundarios (curvas por corrida, inicialización, lote, ejemplos de error) están en el apéndice.
- Dos figuras de assets/ son recortes o reorganizaciones de un gráfico original (sufijos __sesgo y __grilla); FUENTES.json apunta al original.

## Reglas para editar

- Sin texto de relleno: si una línea no agrega un número, una decisión o una cautela, no va.
- Título = la conclusión de la diapositiva, en una línea. La línea inferior (`.take`) solo si dice algo que el título no dice.
- "TP3 · Grupo 4" va solo en la portada; los separadores no llevan encabezado.
- Un gráfico por diapositiva. Debajo, una sola línea de conclusión (`.take`), o hasta tres líneas cortas al costado (`.evidence.side`).
- Lo que no entra en esa línea va a las notas, no a la diapositiva.
- Separador de sección: `<section class="section-divider">` con número, título y la línea `.roadmap` (copiar uno existente).
- Los números salen de reports/ y results/. FUENTES.json identifica el origen de cada gráfico en assets/.
