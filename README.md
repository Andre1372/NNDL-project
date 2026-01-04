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

### 5. Come usare il progetto e fare test iniziali

Il progetto è strutturato in modo che tutte le classi e funzioni vengano definite all'interno di _\src_. Tutti i test e il main code viene scritto all'interno di _\notebooks_. Quindi i notebook sono i main file mentre tutto lo scheletro del progetto è nella cartella _\src_.

Sono già presenti 4 notebooks che sono utili per capire cosa è stato fatto fin ora. Potreste eseguirli e visualizzarli nel seguente ordine:

1. `hardware_test.ipynb` per testare la corretta installazione di pytorch e la potenza di calcolo del computer.
2. `Tutorial_pretty_midi_library.ipynb` fatto da Colin Raffel ed è utile per capire cosa può fare la sua libreria. Dateci uno sguardo veloce, non è troppo importante.
3. `example_audio.ipynb` per vedere come creare il dataset di MIDI files.
4. `example_usage.ipynb` per un esempio della struttura complessiva di training, saving e visualization di un modello Pytorch Lightning.

