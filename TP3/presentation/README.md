# TP3 presentation

reveal.js deck in `index.html`, theme in `style.css`. reveal.js and KaTeX are vendored in `vendor/` so it works offline.

## Present

Open `index.html` in Chrome (double-click works; no server needed).

- `→` / `←`: next / previous slide
- `S`: speaker view with the notes (`<aside class="notes">`)
- `F`: fullscreen
- Figures are linked from `../reports/`, so run the experiments first. Regenerating a figure updates the deck.

## Export to PDF

Open `index.html?print-pdf` in Chrome, print (`Ctrl+P`), destination "Save as PDF", margins "None", background graphics on.

## Structure

Numbered dividers mark the hierarchy:

- `exercise-slide`: one per exercise ("Ejercicio 1", "Ejercicio 2", ...).
- `section-slide`: sections inside an exercise, numbered `<exercise>.<section>` (1.1 EDA, 1.2 Entrenamiento, ...).
- An appendix per exercise goes at the end of that exercise, labelled `<exercise> · Apéndice`.

## Style rules (keep the deck consistent)

- The title states the finding ("El sigmoide llega a un 35 % menos de error"), not the topic ("Resultados").
- One figure per slide, as large as possible. At most 2-3 short lines or a small table next to it.
- Explanations go in the speaker notes, not on the slide.
- Only real numbers from `reports/`, written with a decimal comma (0,888).
- No decoration: no icons, emoji, gradients, shadows, rounded cards or extra colors. The only accent is `--accent` (matplotlib's default blue, to match the figures).
- Reuse the layout classes in `style.css` (`figure`, `split`, `centered-slide`, tables) instead of inline styles.
- Figures use matplotlib's default style; make them in the experiment code, never by hand.
