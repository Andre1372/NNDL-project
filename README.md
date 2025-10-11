# NNDL-project

Progetto per il corso di Neural Networks and Deep Learning.

## Setup

Dopo aver clonato la repository con
```bash
git clone https://github.com/Andre1372/NNDL-project
```

### 1. Creare il virtual environment

```bash
# Creare virtual environment
python -m venv venv_deep

# Attivare virtual environment
# Su Linux/Mac:
source venv_deep/bin/activate
# Su Windows:
venv_deep\Scripts\activate
```

### 2. Installare le dipendenze

```bash
pip install -r requirements.txt
```

## Struttura del progetto

```
NNDL-project/
├── src/              # Codice sorgente del progetto
├── data/             # Dataset
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