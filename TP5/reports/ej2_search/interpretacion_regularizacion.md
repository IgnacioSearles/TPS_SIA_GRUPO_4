# Cómo interpretar la prueba de beta

La KL promediada por entrada no es solamente una medida de cuánto coincide
la nube global de códigos con el prior. Con q(z) como posterior agregado:

\[
\mathbb E_{p_D(x)}D_{KL}(q(z\mid x)\Vert p(z))
=I_q(X;Z)+D_{KL}(q(z)\Vert p(z)).
\]

Por eso, aumentar beta penaliza tanto la información que el código conserva
de cada entrada como el desajuste del agregado respecto del prior. Una KL media
menor puede acompañarse de pérdida de detalle; no demuestra por sí sola una
mejor generación. Esta descomposición aparece en las ecuaciones 1–4 de
[Menon, Blei y Vondrick, UAI 2022](https://proceedings.mlr.press/v180/menon22a/menon22a.pdf).

Es la motivación del control pareado: variar el coeficiente existente y evaluar
reconstrucción, limpieza y generaciones, conservando los términos MSE y KL.
El diagnóstico de muestreo utiliza el agregado empírico de entrenamiento;
sus conclusiones no se trasladan automáticamente a toda la población de imágenes.

Para la exposición puede agregarse una página aclaratoria nueva. No se modifica
ninguna fórmula, texto o nota de las diapositivas existentes.
