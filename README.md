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

Ora è **necessario installare fluidsynth** per riprodurre file audio. Una volta aperta una PowerShell da amministratore eseguire:
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

Per scaricare il dataset da inserire nella cartella /data visitare il link [https://magenta.tensorflow.org/datasets/maestro#v300](https://magenta.tensorflow.org/datasets/maestro#v300) e scaricare la versione 3.0.0.

In seguito estrarre il dataset ed inserirlo nella cartella _/data/maestro-v3.0.0/_.


### 4. Struttura del progetto

Verificate che la seguente struttura del progetto sia uguale alla vostra.

```
NNDL-project/
├── checkpoints/           # Checkpoints del modello durante il training (pytorch)
├── data/                  # Dataset
│   ├── maestro-v3.0.0/        # Clean MIDI subset contiene sottocartelle di .mid
|   ├── processed_npz/         # Conterrà i .npz preprocessati
│   └── prove/                 # Contiene un sottoinsieme di circa 300 .mid
├── lightning_logs/        # PyTorch Lightning logs (TensorBoard)
├── notebooks/             # Jupyter notebooks per esperimenti
├── report/                # Report LaTeX
├── saved_models/          # Modelli salvati al termine del training
├── src/                   # Codice sorgente del progetto (modular structure)
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # PyTorch Lightning model architectures
│   ├── training/              # Training and evaluation with Lightning
│   └── utils/                 # Utility functions
├── venv_deep/             # Virtual environment
├── .gitignore             # File da ignorare nella repository
├── README.md              # Questo file
└── requirements.txt       # Dipendenze Python
```

### 5. Come testare i modelli

Dal file `notebooks/main_testing.ipynb`  è possibile testare i modelli.
Ne lasciamo due caricati nella cartella checkpoints:
 - Best model: best_model.ckpt
 - Last model: last_model.ckpt

