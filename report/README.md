# Report LaTeX

Questa cartella contiene il report del progetto in formato LaTeX.

## Compilazione

Per compilare il documento LaTeX:

```bash
cd report
pdflatex main.tex
pdflatex main.tex  # Eseguire due volte per aggiornare i riferimenti
```

Oppure con latexmk per compilazione automatica:

```bash
cd report
latexmk -pdf main.tex
```

## Struttura

- `main.tex`: File principale del documento
- `figures/`: Cartella per le immagini (da creare se necessario)
