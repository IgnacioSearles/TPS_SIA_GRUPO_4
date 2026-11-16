# Escucha completa de denoising NSynth

Seis ejemplos fijados en results/nsynth_cnn_study/data/manifest.json. Modelos AE densos y convolucionales de results/ej2_nsynth_cnn_dae_{mlp,conv}/base__seed={7,17,27}. No se entrenan ni modifican.

La contaminación se regenera con src.analysis.nsynth_cnn_evaluation.corruption sobre las 224 notas de prueba antes de seleccionar 6. Los 12 hashes de entrada se verifican contra ambos summary.json originales. Las 144 reconstrucciones y 144 entradas tienen MSE por nota comprobada con per_note.csv original, tolerancia 1e-7.

El ruido es gaussiano sobre valores mel normalizados, con recorte [0,1], no ruido acústico añadido a la onda. Todos los audios se regeneran con PGHI y 60 iteraciones Griffin-Lim e inversión mel 200. Una mejora por reconstruir fase no es una mejora de arquitectura. No se reemplazan WAV históricos.

WAV: audio/seed={seed}/sigma={level}/{clean,noisy,dense,cnn}/{label}.wav. Todos mono PCM16, 16000Hz, 1s y normalizados por pico del códec. Sin igualación de sonoridad. La página incorpora todos los audios sin requerir conexión.
