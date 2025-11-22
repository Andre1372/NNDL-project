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

Ora è necessario installare fluidsynth per riprodurre file audio.
Una volta aperta una PowerShell da amministratore eseguire:
```bash
choco install fluidsynth
```
Per verificare la corretta installazione digitare
```bash
fluidsynth -V
```


#### Mac

Forse è possibile sfruttare la GPU integrata nei processori M1, M2, M3, M4 che si chiama mps, quindi cancellare la riga 2 di `requirements.txt`. **Poi ricordarsi di non pushare questa modifica.**
Infine digitare (potrebbe metterci qualche minuto):
```bash
pip install -r requirements.txt
```

Ora è **necessario installare fluidsynth** per riprodurre file audio.


### 3. Download del dataset

Per scaricare il dataset da inserire nella cartella /data visitare il link [https://colinraffel.com/projects/lmd/](https://colinraffel.com/projects/lmd/).

In questa pagina copiare il mirror link relativo a Clean MIDI subset e incollarlo in una nuova pagina.
Poichè la sorgente non è protetta è necessario consentire il download poichè il browser cercherà di bloccarlo.

In seguito estrarre il dataset ed inserirlo nella cartella data.
Dovreste ritrovarvi nella situazione _/data/clean_midi/_ che contiene una serie di sottocartelle contenenti i file .mid.


### 4. Test iniziali

È possibile verificare la potenza di calcolo dell'hardware della macchina runnando il file `hardware_test.py` nella cartella src.

Mentre attraverso il file `example_usage.ipynb` si può testare la corretta installazione dei pacchetti.
Questo file eseguirà il training di una semplice rete neurale feedforward per risolvere il problema di regressione.



### 5. Monitoraggio del training con TensorBoard

PyTorch Lightning salva automaticamente i log del training. Per visualizzarli eseguire dal terminale il comando:
PyTorch Lightning salva automaticamente i log del training. Per visualizzarli eseguire dal terminale il comando:
```bash
tensorboard --logdir=lightning_logs/
```


## Struttura del progetto

```
NNDL-project/
├── src/                    # Codice sorgente del progetto (modular structure)
│   ├── config/            # Configuration management
│   ├── config/            # Configuration management
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # PyTorch Lightning model architectures
│   ├── training/          # Training and evaluation with Lightning
│   ├── utils/             # Utility functions
│   ├── utils/             # Utility functions
│   └── README.md          # Documentazione dettagliata della struttura src/
├── data/                   # Dataset
├── checkpoints/            # Checkpoints del modello durante il training (pytorch)
├── saved_models/           # Modelli salvati al termine del training
├── lightning_logs/         # PyTorch Lightning logs (TensorBoard)
├── checkpoints/            # Checkpoints del modello durante il training (pytorch)
├── saved_models/           # Modelli salvati al termine del training
├── lightning_logs/         # PyTorch Lightning logs (TensorBoard)
├── notebooks/              # Jupyter notebooks per esperimenti
├── report/                 # Report LaTeX
└── requirements.txt        # Dipendenze Python
```

Per informazioni dettagliate sulla struttura modulare del codice sorgente, consultare [src/README.md](src/README.md).


## Report

Il report del progetto è disponibile nella cartella `report/` in formato LaTeX.
Il main file è `template.tex` e contiene tutto lo scheletro del report.

