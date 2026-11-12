# Ejercicio 1a: búsqueda de la configuración del autoencoder

**Objetivo (consigna):** aprender las 32 letras de 5×7 con latente 2 y a lo sumo 1 píxel mal
por letra. Se buscó la mejor combinación de arquitectura, activación, inicializador,
optimizador, learning rate (η), pérdida y batch, sin suponer nada sin una razón teórica o
medida. Tablas completas: `etapa1.md` y `etapa2_<prueba>.md`. Razonamiento mas detallado en `bitacora.md`.

## Método

10 semillas por configuración, 10000 épocas y un control cada 10 épocas (en cada control se
miden los píxeles mal de las 32 letras). Las configuraciones se ordenan por:

1. **Semillas que terminan en la meta** (una corrida que diverge cuenta como falla).
2. **Recaídas:** porcentaje de controles fuera de la meta después de alcanzarla por primera vez,
   redondeado al 1 % para que un control aislado no pese más que una diferencia de velocidad.
3. **Pasos hasta la meta (mediana).** Un paso es una actualización de los pesos. Con batch 32
   (las 32 letras juntas), un paso por época.
4. **Menos parámetros** (pesos + sesgos) como desempate.

Con 10 semillas, la tasa de éxito solo distingue diferencias grandes, así que el criterio 1
descarta y los pasos son los que separan.

## Resultados

**Etapa 1: búsqueda amplia (2160 corridas).** Anchos 16 a 128 × 1 a 3 capas × η 0.01, 0.003 y
0.001 × tanh, relu, leaky_relu y sigmoid, cada una con su inicializador teórico (y sigmoid con
tres). Optimizador Adam.
- **El ancho es lo que importa.** Con una capa cumplen 119 de 180 semillas con ancho 16,
  178 con 32 y 180 con 64 y con 128: pasado cierto ancho, más capacidad no agrega nada.
- **tanh, relu y leaky_relu empatan** en velocidad y confiabilidad.
- **Sigmoid se descarta:** aun con la inicialización corregida para ella (xavier × 4) es 2,85×
  más lenta que tanh. Es consistente con que su salida no está centrada en 0.

**Etapa 2: cambiar una cosa por vez (1890 corridas).** Sobre las redes 64, 64-32 y 128-64 con
tanh, relu y leaky_relu. Solo se cruzaron pares con interacción conocida (optimizador × η,
batch × η).

| Prueba | Resultado | Elección |
|---|---|---|
| η | 0.03 es más rápido pero cerca del límite (128-64 pierde una semilla); 0.1 falla casi siempre | **0.01**, por margen de estabilidad |
| Optimizador | Momentum es 2 a 3× más lento y diverge; RMSprop tiene 1 a 9 % de recaídas | **Adam** |
| Inicializador | Con tanh, xavier es claramente la más rápida; con relu, he y xavier empatan | **El emparejado** (xavier con tanh, he con relu) |
| Pérdida | MSE: 0/90. Los píxeles mal quedan saturados en el valor opuesto y el gradiente se anula | **BCE** |
| Batch | Con η 0.01 los batches chicos se vuelven inestables (batch 4: 33/90); nunca superan a 32 | **32** |

**Profundidad contra angostura (360 corridas).** Con capas que se achican a la mitad, la
tercera capa era también la más angosta. Separando las dos cosas con ancho 16: **relu falla
por angostura** (una capa de 4 baja de 37/40 a 14/40, p < 0,0001, con neuronas muertas en parte
de las semillas). **tanh falla por profundidad** (3 capas: de 16/20 a 7/20, p = 0,01). Con ancho
64 o más no aparece ninguno de los dos efectos.

**relu 32 (210 corridas).** Con las mismas pruebas da las mismas conclusiones, pero tolera peor
los cambios: con small_normal o con momentum y η bajo cae a 5/10, donde las redes de ancho 64
cumplían 10/10.

## Candidatas finales

Todas con el inicializador emparejado, Adam con η 0.01, BCE, batch 32 y latente 2. Todas
cumplen **10/10 sin recaídas relevantes**. Tiempos medidos en un solo núcleo.

| Red | Parámetros | Pasos hasta la meta (mediana, Q1–Q3) | Entrenamiento hasta la meta | Inferencia: 1 letra / 32 |
|---|---|---|---|---|
| relu 32 | **2469** | 395 (378–575) | 70 ms | 12 µs / **36 µs** |
| tanh 64 | 4901 | 245 (230–260) | **55 ms** | 12 µs / 85 µs |
| tanh 64-32 | 8933 | 175 (168–188) | 60 ms | 19 µs / 112 µs |
| tanh 128-64 | 26021 | **135 (130–142)** | 67 ms | 21 µs / 204 µs |

*Entrenamiento hasta la meta* = pasos hasta la meta × tiempo por paso. *Inferencia* = usar la
red ya entrenada (codificar y decodificar) con 1 letra o con las 32 juntas.

- **relu 32, la más simple:** la menor cantidad de parámetros y la inferencia más barata.
- **tanh 64, la más robusta:** con ancho 64 cumplió con todas las activaciones, η e
  inicializadores probados. Además es la que menos tarda en entrenar en tiempo real.
- **tanh 64-32, el equilibrio:** intermedia en todo. No gana en ningún criterio por sí sola.
- **tanh 128-64:** gana solo en pasos. Cada paso cuesta más del doble, así que no gana en tiempo.

Menos pasos no significa menos tiempo: las cuatro entrenan en 55 a 70 ms y hacen inferencia en
microsegundos. Elegir entre ellas es priorizar simplicidad, robustez o velocidad en pasos.

## Limitaciones

- La etapa 2 cambia una cosa por vez, así que dos cambios combinados podrían rendir distinto.
- No se probó η entre 0.03 y 0.1.
- Los mecanismos de la sección de profundidad son hipótesis con evidencia parcial.
- Se entrena y se evalúa con las mismas 32 letras: "mejor" quiere decir que memoriza mejor
  (lo que pide la consigna), no que generaliza.

## Reproducir (desde `TP5`)

```bash
python main.py configs/ej1a_search_*.json --jobs 10    # etapa 1
python main.py configs/ej1a_etapa2/*.json --jobs 10    # etapa 2, profundidad y relu 32
python scripts/ej1a_search_table.py [--stage 2]        # tablas
```
