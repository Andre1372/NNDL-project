# NNDL-project

Progetto per il corso di Neural Networks and Deep Learning.

## Setup

### 1. Creare il virtual environment

```bash
# Creare virtual environment
python -m venv venv

# Attivare virtual environment
# Su Linux/Mac:
source venv/bin/activate
# Su Windows:
venv\Scripts\activate
```

### 2. Installare le dipendenze

```bash
pip install -r requirements.txt
```

## Struttura del progetto

```
NNDL-project/
├── src/              # Codice sorgente del progetto
├── data/             # Dataset (non versionato)
├── notebooks/        # Jupyter notebooks per esperimenti
├── models/           # Modelli salvati
├── report/           # Report LaTeX
└── requirements.txt  # Dipendenze Python
```

## Tecnologie utilizzate

- **PyTorch**: Framework per deep learning
- **NumPy/Pandas**: Elaborazione dati
- **Matplotlib**: Visualizzazione
- **Jupyter**: Ambiente di sviluppo interattivo

## Report

Il report del progetto è disponibile nella cartella `report/` in formato LaTeX.