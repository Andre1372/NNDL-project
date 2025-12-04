
# Base Dataset class from PyTorch
from torch.utils.data import Dataset
# For transformations on data
from typing import Optional, Callable
# For file operations
from pathlib import Path
# For midi processing
import pretty_midi
# To work with matrices
import numpy as np
import scipy.sparse as sparse
import torch
# For progress bars
from tqdm import tqdm
# For garbage collection
import gc

class PolyDataset(Dataset):

    def __init__(self, inputs: np.ndarray, targets: np.ndarray, transform: Optional[Callable] = None):
        """
        Initialize the dataset.
        
        Args:
            inputs: Input features
            targets: Target values
            transform: Optional transform to apply to inputs
        """
        self.data = [(x, y) for x, y in zip(inputs, targets)]
        self.transform = transform

    def __len__(self) -> int:
        """ Return the number of samples in the dataset. """
        return len(self.data)

    def __getitem__(self, idx: int):
        """ Get a sample by index. """
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

class PianoDataset(Dataset):

    def __init__(self, file_paths, transform: Optional[Callable] = None):
        """
        Initialize the dataset.
        
        Args:
            file_paths: List of file paths to .npz files
            transform: Optional transform to apply to inputs
        """
        self.file_paths = file_paths
        self.transform = transform

        self.bars_per_segment = 8
        self.steps_per_bar = 16
        self.pitch_dim = 128

    def __len__(self) -> int:
        """ Returns the total number of (prev, curr) pairs available. """
        return len(self.file_paths) * self.bars_per_segment

    def __getitem__(self, idx: int):
        """ Get a sample by index. """

        # 1. Identifica quale file e quale battuta (0-7) stiamo cercando
        file_idx = idx // self.bars_per_segment
        bar_idx = idx % self.bars_per_segment
                
        # Nota: caricarla ogni volta può essere lento (Disk I/O). 
        # Per iniziare va bene, per ottimizzare in futuro caricheremo tutto in RAM.
        sparse_matrix = sparse.load_npz(self.file_paths[file_idx])
        dense_matrix = sparse_matrix.todense() # Shape: (128, 128)
        
        # 2. Estrai la Current Bar (Target)
        # Affettiamo sull'asse del tempo (asse 0)
        start_t = bar_idx * self.steps_per_bar
        end_t = start_t + self.steps_per_bar
        
        # current_bar shape: (128, 16)
        current_bar = dense_matrix[:, start_t:end_t]

        # 3. Estrai la Previous Bar (Condition)
        if bar_idx == 0:
            # Se è la prima battuta del segmento, la precedente è "vuota" (padding)
            prev_bar = np.zeros((self.pitch_dim, self.steps_per_bar), dtype=np.float32)
        else:
            # Altrimenti prendiamo i 16 step precedenti
            prev_start = (bar_idx - 1) * self.steps_per_bar
            prev_end = prev_start + self.steps_per_bar
            prev_bar = dense_matrix[:, prev_start:prev_end]

        # 4. Trasformazioni (es. ToTensor)
        if self.transform:
            current_bar = self.transform(current_bar)
            prev_bar = self.transform(prev_bar)

        # MidiNet richiede anche una dimensione canale: (1, 16, 128)
        return prev_bar, current_bar
    
class PianoMmapDataset(Dataset):
    def __init__(self, mmap_path: str, split_indices: np.ndarray, transform=None):
        """
        Initialize the dataset.
        Args:
            mmap_path: Path al file unico .dat
            split_indices: Array di indici (interi) che corrispondono ai segmenti 
                           assegnati a questo split (es. solo i segmenti di training).
            transform: Trasformazioni opzionali (es. ToTensor)
        """
        self.mmap_path = mmap_path
        self.split_indices = split_indices  # Questa è la "maschera" di accesso
        self.transform = transform
        
        # Costanti
        self.bars_per_segment = 8
        self.steps_per_bar = 16
        self.pitch_dim = 128
        
        # Apriamo il file in modalità lettura.
        # Importante: shape[0] non è len(split_indices), ma la dimensione TOTALE del file su disco.
        # Dobbiamo sapere quanti segmenti totali ci sono nel file .dat (es. 500000).
        # Per semplicità qui assumo che tu lo sappia o lo calcoli dalla dimensione file.
        # Esempio: total_segments = os.path.getsize(path) / (128*128)
        
        # TIP: Passare total_num_segments come argomento o calcolarlo è necessario per il memmap
        # Qui lo calcolo dinamicamente per robustezza:
        import os
        file_size = os.path.getsize(mmap_path)
        self.total_segments_on_disk = file_size // (128 * 128) # 1 byte per elemento (uint8)
        
        self.data = np.memmap(mmap_path, dtype='uint8', mode='r', 
                              shape=(self.total_segments_on_disk, 128, 128))

    def __len__(self) -> int:
        # Il dataset è grande quanto il numero di segmenti nello split * 8 battute
        return len(self.split_indices) * self.bars_per_segment

    def __getitem__(self, idx: int):
        # 1. Mappatura Logica -> Fisica
        # L'idx del dataloader va da 0 a len(self).
        # Dobbiamo capire a quale indice della lista `split_indices` corrisponde.
        list_idx = idx // self.bars_per_segment  # Indice nella lista split_indices
        bar_idx = idx % self.bars_per_segment    # Quale battuta (0-7)
        
        # Ora recuperiamo il VERO indice sul disco
        physical_segment_idx = self.split_indices[list_idx]
        
        # 2. Accesso ai dati (Memory Mapped)
        # Leggiamo solo la riga specificata da physical_segment_idx
        full_segment = np.array(self.data[physical_segment_idx], dtype=np.float32)
        
        # 3. Slicing (Battuta Corrente e Precedente)
        start_t = bar_idx * self.steps_per_bar
        end_t = start_t + self.steps_per_bar
        
        current_bar = full_segment[:, start_t:end_t]

        if bar_idx == 0:
            # Padding vuoto per la prima battuta
            prev_bar = np.zeros((self.pitch_dim, self.steps_per_bar), dtype=np.float32)
        else:
            prev_start = (bar_idx - 1) * self.steps_per_bar
            prev_end = prev_start + self.steps_per_bar
            prev_bar = full_segment[:, prev_start:prev_end]

        # 4. ToTensor
        curr_tensor = torch.from_numpy(current_bar).unsqueeze(0)
        prev_tensor = torch.from_numpy(prev_bar).unsqueeze(0)
        
        if self.transform:
            curr_tensor = self.transform(curr_tensor)
            prev_tensor = self.transform(prev_tensor)

        return prev_tensor, curr_tensor

class MidiPreprocessor:
    def __init__(self, select_instruments: list, note_start: int, note_end: int, output_dir: Path):
        """
        Initialize the MIDI preprocessor. It will divide MIDI files into piano roll segments of 8 bars (128 time steps).
        Args:
            select_instruments: List of MIDI program numbers to select (e.g., [0] for Acoustic Grand Piano)
            note_start: The starting MIDI note number (inclusive)
            note_end: The ending MIDI note number (exclusive)
            output_dir: Directory to save processed files
        """
        self.select_instruments = select_instruments
        self.note_start = note_start
        self.note_end = note_end
        self.output_dir = output_dir

    # To let this class be called like a function
    def __call__(self, midi_file_path: Path):

        # Create unique ID combining Artist and Title since more artists can have same song titles
        artist = midi_file_path.parent.name.replace(" ", "_")
        song_name = midi_file_path.stem.replace(" ", "_")
        unique_id = f"{artist}__{song_name}"

        try:
            result = []

            # Extract piano tracks
            pm = pretty_midi.PrettyMIDI(str(midi_file_path))

            # Extract only selected instruments
            piano_instruments = [instr for instr in pm.instruments if instr.program in self.select_instruments and not instr.is_drum]
            if piano_instruments == []:
                return f"DISCARDED: {unique_id}: NO PIANO" # Skip files with no piano instruments
            
            # Extract tempo changes
            tempo_change_times, tempi = pm.get_tempo_changes()

            for i in range(len(tempo_change_times)):
                # Find start and end times for this tempo segment
                t_start = tempo_change_times[i]
                t_end = tempo_change_times[i+1] if i+1 < len(tempo_change_times) else pm.get_end_time()
                
                bpm = tempi[i]
                if bpm <= 0: continue # Skip invalid bpm

                # Compute fs to have exactly 16 notes every 4 beats (4/4 time signature)
                # one beat tempo = bpm/60  -->  16 fs = 4 bpm / 60  -->  fs = bpm/15 
                fs = bpm / 15.0 
                
                # Create the exact time grid for this segment
                times = np.arange(t_start, t_end, 1./fs)
                
                # If the segment is too short (less than 8 bars or less than 128 samples), skip it
                SAMPLES_PER_8_BARS = 128 
                if len(times) < SAMPLES_PER_8_BARS:
                    continue

                programs_in_segment = []

                for instr in piano_instruments:
                    if instr.program in programs_in_segment:
                        continue  # Skip duplicate instruments
                    programs_in_segment.append(instr.program)

                    # Compute the piano roll only for the specified times (128, len(times))
                    piano_roll = instr.get_piano_roll(fs=fs, times=times) 
                    
                    # Binarization (Velocity > 0 becomes 1)
                    piano_roll[piano_roll > 0] = 1
                    
                    # Ignore notes outside the specified range
                    piano_roll[:self.note_start, :] = 0
                    piano_roll[self.note_end:, :] = 0

                    # Segment the matrix into chunks of width 128 (8 bars)
                    num_windows = piano_roll.shape[1] // SAMPLES_PER_8_BARS
                    
                    for w in range(num_windows):
                        start_col = w * SAMPLES_PER_8_BARS
                        end_col = start_col + SAMPLES_PER_8_BARS
                        
                        window = piano_roll[:, start_col:end_col]
                        
                        if np.sum(window) == 0:
                            continue  # Skip empty windows

                        if window.shape[1] != SAMPLES_PER_8_BARS or window.shape[0] != 128:
                            continue
                            
                        # Save an efficient sparse representation
                        output_filename = f"{unique_id}_instr{instr.program}_ts{times[start_col]:.2f}_te{times[end_col-1]:.2f}_fs{fs:.2f}.npz"
                        save_path = self.output_dir / output_filename
                        sparse_window = sparse.csr_matrix(window)
                        sparse.save_npz(save_path, sparse_window)
                        result.append({"filename": output_filename, "instrument": instr.program, "fs": fs, "bpm": bpm})

            if len(result) == 0:
                return f"DISCARDED: {unique_id}: NO VALID SEGMENTS"
            
            return result
        
        except Exception as e:
            return f"ERROR: {unique_id}: {e}"
        
def create_mmap_dataset(source_dir: Path, output_file: Path):
    """ 
    Consolidate all the files .npz in one memmap file.
    Args:
        source_dir (Path): Directory containing the .npz files.
        output_file (Path): Path to the output memmap file.
    """
    source_dir = Path(source_dir)
    output_file = Path(output_file)
    
    # If the file exists, try to remove it
    if output_file.exists():
        try:
            output_file.unlink()
        except OSError as e:
            raise RuntimeWarning(f"Unable to remove existing file. It might be in use. Error: {e}")

    files = sorted(list(source_dir.glob("*.npz")))
    num_files = len(files)
    
    if num_files == 0:
        raise ValueError("No .npz files found.")
    
    # Create the memmap file
    fp = np.memmap(output_file, dtype='uint8', mode='w+', shape=(num_files, 128, 128))

    try:
        for i, file_path in enumerate(tqdm(files, ncols=150, desc="Consolidating .npz files")):
            try:
                sparse_matrix = sparse.load_npz(file_path)
                dense_matrix = sparse_matrix.todense()
                fp[i] = dense_matrix.astype('uint8')
            except Exception as e:
                print(f"Error in file {file_path}: {e}")
                fp[i] = np.zeros((128, 128), dtype='uint8')
        
        # Flush data to disk
        fp.flush()
        
    finally:
        # Delete the Python object and force garbage collector to release the file handle
        del fp
        gc.collect()