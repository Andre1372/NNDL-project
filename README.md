# NNDL-project

Progetto per il corso di Neural Networks and Deep Learning.

## Setup

Dopo aver clonato la repository con:
```bash
git clone https://github.com/Andre1372/NNDL-project
```

### 1. Creare il virtual environment

#### Windows 
```bash
# Creare virtual environment
python -m venv venv_deep

# Attivare virtual environment
venv_deep\Scripts\activate

# Verificare l'attivazione
Get-Command python
```

#### Mac/Linux
```bash
# Creare virtual environment
python3 -m venv venv_deep

# Attivare virtual environment
source venv_deep/bin/activate

# Verificare l'attivazione
command -v python3
```

Prima di continuare è necessario associare il virtual environment `venv_deep` alla cartella su VS Code. Per farlo, apri il Command Palette con `Ctrl + Shift + P` (Windows/Linux) o `Cmd + Shift + P` (macOS). Digita `Python: Select Interpreter` e seleziona l'environment appena creato.

### 2. Installare le dipendenze

Per il progetto sono necessarie le seguenti estensioni di VS Code:
- Python
- Jupyter
- LaTeX Workshop

#### Windows
Prima di tutto verificare che versione cuda si possiede sulla GPU NVIDIA attraverso il comando `nvidia-smi`. In seguito sul sito `https://pytorch.org/get-started/locally/` si può ottenere il comando completo del tipo:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
In questo caso sostituire la riga 2 di `requirements.txt` con:
```bash
--index-url https://download.pytorch.org/whl/cu126
```
Infine digitare (potrebbe metterci qualche minuto):
```bash
pip install -r requirements.txt
```

#### Mac

Forse è possibile sfruttare la GPU integrata nei processori M1, M2, M3, M4 che si chiama mps, quindi cancellare la riga 2 di `requirements.txt`. **Poi ricordarsi di non pushare questa modifica.**
Infine digitare (potrebbe metterci qualche minuto):
```bash
pip install -r requirements.txt
```

### 3. Test per verificare l'installazione

È possibile verificare il setup del progetto runnando il file `hardware_test.py` per testare la potenza di calcolo dell'hardware.

```bash
python src/hardware_test.py
```

### 4. Esempio di utilizzo

Per vedere come utilizzare la nuova struttura modulare del progetto, esegui lo script di esempio:

```bash
python example_usage.py
```

Questo script dimostra come:
- Gestire le configurazioni
- Creare e addestrare modelli
- Valutare le performance
- Visualizzare i risultati

## Struttura del progetto

```
NNDL-project/
├── src/                    # Codice sorgente del progetto (modular structure)
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # PyTorch model architectures
│   ├── training/          # Training and evaluation
│   ├── utils/             # Utility functions (logging, metrics, visualization)
│   ├── config/            # Configuration management
│   └── README.md          # Documentazione dettagliata della struttura src/
├── configs/                # Configuration files json
├── data/                   # Dataset
├── models/                 # Modelli salvati
├── notebooks/              # Jupyter notebooks per esperimenti
├── report/                 # Report LaTeX
├── results/                # Results of simulations (images)
└── requirements.txt        # Dipendenze Python
```

Per informazioni dettagliate sulla struttura modulare del codice sorgente, consultare [src/README.md](src/README.md).

## Report

Il report del progetto è disponibile nella cartella `report/` in formato LaTeX.
Il main file è `template.tex` e contiene tutto lo scheletro del report.