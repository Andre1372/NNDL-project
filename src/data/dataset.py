
# Standard lybrary
from pathlib import Path
from typing import List, Dict, Optional, Callable
import gc
# For midi processing
import pretty_midi
# Numpy/torch
import numpy as np
import torch
from torch.utils.data import Dataset
# For dataframes
import pandas as pd
# For progress bars
from tqdm import tqdm
# For sparse matrices
from scipy import sparse

# Constants
SAMPLES_PER_SEGMENT = 128
SAMPLES_PER_BAR = 16
BARS_PER_SEGMENT = SAMPLES_PER_SEGMENT // SAMPLES_PER_BAR
PITCH_DIM = 128


def identify_chord(piano_roll_bar: np.ndarray) -> int:
    """
    Identifica l'accordo dominante in una sezione di piano roll (es. una battuta).
    Ritorna un indice:
    0-11: Major chords (C, C#, D...)
    12-23: Minor chords (Cm, C#m, Dm...)
    24: No Chord / Silence
    """
    # 1. Calcola il Chroma Vector (somma tutta l'energia su 12 note)
    # piano_roll_bar shape: (128, time_steps)
    chroma = np.zeros(12)
    for i in range(128):
        note_idx = i % 12
        chroma[note_idx] += np.sum(piano_roll_bar[i, :])
    
    if np.sum(chroma) == 0:
        return 24  # Silence
    
    # Normalizza
    chroma = chroma / (np.max(chroma) + 1e-6)

    # 2. Definisci i template per Major e Minor
    # C Major: C (1), E (1), G (1) -> indici 0, 4, 7
    # C Minor: C (1), Eb (1), G (1) -> indici 0, 3, 7
    major_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    minor_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

    max_score = -1
    best_chord = 24

    # 3. Confronta con tutti i 12 shift (trasposizioni)
    for root in range(12):
        # Shift circolare del chroma per allinearlo alla radice C
        shifted_chroma = np.roll(chroma, -root)
        
        # Punteggio semplice: dot product
        score_maj = np.dot(shifted_chroma, major_template)
        score_min = np.dot(shifted_chroma, minor_template)

        if score_maj > max_score:
            max_score = score_maj
            best_chord = root  # 0-11
        
        if score_min > max_score:
            max_score = score_min
            best_chord = root + 12 # 12-23

    return best_chord


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

    def __init__(self, data_dir: Path, split_indices: Optional[List[int]] = None, transform: Optional[Callable] = None):
        """
        Dataset that loads piano roll segments from a directory of .npz files.
        ALL bars are loaded into memory during initialization.
        
        Args:
            data_dir: Path to the directory containing .npz files.
            split_indices: Optional list/array of indices to select specific files from the sorted directory listing.
            transform: Optional transformations to apply to the tensors.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load all file paths and sort them to ensure consistent ordering
        all_files = sorted(list(self.data_dir.glob("*.npz")))
        if len(all_files) == 0:
            raise ValueError("No .npz files found.")
        
        # Filter files if split_indices are provided
        if split_indices is not None:
            self.file_paths = [all_files[idx] for idx in split_indices]
        else:
            self.file_paths = all_files
            
        if not self.file_paths:
            print("Warning: PianoDataset initialized with empty file list.")

        # Pre-load all data
        self.data = []
        for file_path in tqdm(self.file_paths, desc="Loading PianoDataset"):
            try:
                sparse_matrix = sparse.load_npz(file_path)
                full_segment = sparse_matrix.toarray().astype(np.float32) # Shape: (128, 128)
                
                # Verify shape
                if full_segment.shape != (PITCH_DIM, SAMPLES_PER_SEGMENT):
                    print(f"Skipping {file_path}: Invalid shape {full_segment.shape}")
                    continue

                # Split directly into bars and store
                self.data.extend([full_segment[:, b*SAMPLES_PER_BAR : (b + 1)*SAMPLES_PER_BAR] for b in range(BARS_PER_SEGMENT)])
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def __len__(self) -> int:
        """Return the total number of bars in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Get a sample (previous bar, current bar) by index."""
        current_bar = self.data[idx]
        
        # Determine if this bar is the start of a segment (every 8th bar)
        is_first_bar = (idx % BARS_PER_SEGMENT == 0)

        if is_first_bar:
            prev_bar = np.zeros((PITCH_DIM, SAMPLES_PER_BAR), dtype=np.float32)
        else:
            prev_bar = self.data[idx - 1]

        # Convert to tensors, unsqueeze(0) adds the channel dimension (1, 128, 16)
        curr_tensor = torch.from_numpy(current_bar).unsqueeze(0)
        prev_tensor = torch.from_numpy(prev_bar).unsqueeze(0)
        
        if self.transform:
            curr_tensor = self.transform(curr_tensor)
            prev_tensor = self.transform(prev_tensor)

        return prev_tensor, curr_tensor


class MidiPreprocessor:
    """
    Preprocessor for MIDI files. 
    It segments MIDI files into piano rolls of fixed length (default 8 bars / 128 samples).
    Results are stored in-memory in `self.bars` and metadata is updated in the provided DataFrame.
    """

    def __init__(self, output_dir: Path, select_instruments: List[int], note_start: int, note_end: int, min_notes: int = 5, min_polyphony: float = 1.0):
        """
        Initialize the MIDI preprocessor.

        Args:
            output_dir: Directory where processed .npz segments will be saved.
            select_instruments: List of MIDI program numbers to select.
            note_start: The starting MIDI note number (inclusive).
            note_end: The ending MIDI note number (exclusive).
            min_notes: Minimum number of notes required in a segment to be kept.
            min_polyphony: Minimum average polyphony required in a segment to be kept.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.select_instruments = select_instruments
        self.note_start = note_start
        self.note_end = note_end
        self.min_notes = min_notes
        self.min_polyphony = min_polyphony

    def _process_piano_roll(self, instr: pretty_midi.Instrument, fs: float, time_segment: np.ndarray) -> np.ndarray:
        """Extracts and cleans the piano roll for a specific instrument and time segment."""
        # Get piano roll (shape: pitch x time)
        piano_roll = instr.get_piano_roll(fs=fs, times=time_segment)
        
        # Binarize: Any velocity > 0 is treated as Note On (1)
        piano_roll = piano_roll.astype(np.float32) / 127.0        
        # Clip pitch range
        piano_roll[:self.note_start, :] = 0
        piano_roll[self.note_end:, :] = 0
        
        return piano_roll

    def _is_segment_valid(self, piano_roll: np.ndarray) -> bool:
        """Checks if a segment is valid based on dimensions, note count, and polyphony."""
        # Check dimensions
        if piano_roll.shape != (PITCH_DIM, SAMPLES_PER_SEGMENT): return False

        # Consideriamo una nota "attiva" se la velocity > 0
        active_notes_mask = piano_roll > 0

        # Check polyphony (Avoid monophonic tracks)
        notes_per_step = np.sum(piano_roll, axis=0)
        active_steps = notes_per_step[notes_per_step > 0]
        avg_polyphony = np.mean(active_steps) if len(active_steps) > 0 else 0
        return avg_polyphony >= self.min_polyphony

    def __call__(self, midi_file_path: Path):
        """
        Process a single MIDI file.
        Returns a list of metadata dictionaries for valid segments or an error string.
        Segments are saved as .npz files.
        """
        # Unique identifier for the file (using parent folder name to avoid collisions)
        file_name = f"{midi_file_path.parent.stem}/{midi_file_path.stem}"
        
        results_meta = []

        try:
            pm = pretty_midi.PrettyMIDI(str(midi_file_path))

            # Filter instruments
            piano_instruments = [instr for instr in pm.instruments if instr.program in self.select_instruments and not instr.is_drum]
            if len(piano_instruments) == 0:
                return f"DISCARDED: {file_name} (NO PIANO)" # Skip files with no piano instruments
            
            # ----------------------------------------------------------
            # 1. Divide the MIDI file into windows of same tempo
            # ----------------------------------------------------------
            # Get tempo changes
            tempo_change_times, tempi = pm.get_tempo_changes()

            segment_global_idx = 0

            for i, t_start in enumerate(tempo_change_times):
                t_end = tempo_change_times[i+1] if i + 1 < len(tempo_change_times) else pm.get_end_time()
                
                bpm = tempi[i]
                if bpm <= 0: continue # Skip invalid bpm

                # Compute fs to have exactly 16 notes every 4 beats (4/4 time signature)
                fs = bpm / 15.0  # 16 fs = 4 bpm / 60 
                
                # Generate time grid
                time_window = np.arange(t_start, t_end, 1.0/fs)
                # Check minimum length
                if len(time_window) < SAMPLES_PER_SEGMENT: continue

                # ----------------------------------------------------------
                # 2. Divide each tempo window into segments of 8 bars
                # ----------------------------------------------------------
                for segment_start_idx in range(0, len(time_window), SAMPLES_PER_SEGMENT):
                    time_segment = time_window[segment_start_idx : segment_start_idx + SAMPLES_PER_SEGMENT]
                    
                    # ----------------------------------------------------------
                    # 3. Process each instrument for this segment
                    # ----------------------------------------------------------
                    for instr in piano_instruments:
                        piano_roll = self._process_piano_roll(instr, fs, time_segment)

                        if not self._is_segment_valid(piano_roll): continue

                        # --- ### MODIFICA: ESTRAZIONE ACCORDI ---
                        # Dividiamo il segmento (128 steps) in 8 battute da 16 steps
                        segment_chords = []
                        steps_per_bar = 16
                        for b in range(8):
                            b_start = b * steps_per_bar
                            b_end = (b + 1) * steps_per_bar
                            # Estraiamo la porzione di matrice corrispondente alla battuta
                            bar_roll = piano_roll[:, b_start:b_end]
                            # Identifichiamo l'accordo
                            chord_idx = identify_chord(bar_roll)
                            segment_chords.append(chord_idx)
                        
                        segment_chords = np.array(segment_chords, dtype=np.uint8)
                        # ----------------------------------------

                        # Save segment to disk
                        save_name = f"{midi_file_path.parent.stem}_{midi_file_path.stem}_{segment_global_idx}.npz"
                        save_path = self.output_dir / save_name
                        
                        # --- ### MODIFICA: SALVATAGGIO COMPRESSO ---
                        # Invece di sparse.save_npz, usiamo np.savez_compressed
                        # per salvare sia il piano roll (come sparse) che gli accordi.
                        sparse_matrix = sparse.csr_matrix(piano_roll.astype(np.float32))
                        
                        np.savez_compressed(
                            save_path, 
                            piano_roll=sparse_matrix, # Chiave 'piano_roll'
                            chords=segment_chords,    # Chiave 'chords'
                            bpm=bpm
                        )
                        # -------------------------------------------

                        # Store metadata
                        results_meta.append({
                            'filename': save_name,
                            'original_file': file_name,
                            'instrument': instr.program,
                            'bpm': bpm,
                            'fs': fs
                        })
                        
                        segment_global_idx += 1

            if segment_global_idx == 0:
                return f"DISCARDED: {file_name} (NO VALID SEGMENTS)"
            
            return results_meta

        except Exception as e:
            return f"ERROR: {file_name}: {str(e)}"
        
# =================================================================================
# PART 3: MEMORY MAPPED DATASET & UTILS (Add this to the end of dataset.py)
# =================================================================================

def create_mmap_dataset(source_dir: Path, output_prefix: str):
    """ 
    Consolidate .npz files into two memmap files: one for piano rolls, one for chords.
    Args:
        source_dir (Path): Directory containing the .npz files.
        output_prefix (str): Prefix for output files (e.g. 'data/train' -> data/train_piano.dat, data/train_chords.dat)
    """
    source_dir = Path(source_dir)
    piano_out = Path(f"{output_prefix}_piano.dat")
    chords_out = Path(f"{output_prefix}_chords.dat")
    
    # Clean up existing
    for p in [piano_out, chords_out]:
        if p.exists():
            try:
                p.unlink()
            except OSError as e:
                print(f"Warning: Could not delete {p}: {e}")

    files = sorted(list(source_dir.glob("*.npz")))
    num_files = len(files)
    
    if num_files == 0:
        raise ValueError("No .npz files found in source_dir.")
    
    print(f"Found {num_files} segments. Creating memory mapped files...")

    # 1. Create memmap placeholders
    # Piano: (Num_Segments, 128 pitch, 128 time)
    fp_piano = np.memmap(piano_out, dtype='float32', mode='w+', shape=(num_files, PITCH_DIM, SAMPLES_PER_SEGMENT))
    # Chords: (Num_Segments, 8 bars) - stores integer indices 0-24
    fp_chords = np.memmap(chords_out, dtype='uint8', mode='w+', shape=(num_files, BARS_PER_SEGMENT))

    try:
        # 2. Fill them
        for i, file_path in enumerate(tqdm(files, ncols=80, desc="Consolidating")):
            try:
                # Load compressed file
                with np.load(file_path, allow_pickle=True) as data:
                    # Piano Roll (saved as sparse csr inside the npz)
                    sparse_matrix = data['piano_roll'].item() 
                    fp_piano[i] = sparse_matrix.todense().astype('float32')
                    
                    # Chords (saved as dense array)
                    fp_chords[i] = data['chords'].astype('uint8')
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                # Fill with zeros/silence on error to keep alignment
                fp_piano[i] = np.zeros((PITCH_DIM, SAMPLES_PER_SEGMENT), dtype='float32')
                fp_chords[i] = np.full((BARS_PER_SEGMENT,), 24, dtype='uint8') # 24 = Silence code
        
        # Flush changes to disk
        fp_piano.flush()
        fp_chords.flush()
        print(f"Dataset created successfully:\n- {piano_out}\n- {chords_out}")
        
    finally:
        # Release file handles
        del fp_piano
        del fp_chords
        gc.collect()


class PianoMmapDataset(Dataset):
    def __init__(self, mmap_prefix: str, split_indices: np.ndarray, transform=None):
        """
        Dataset that reads from memory-mapped files (.dat).
        Returns triplets: (prev_bar, current_bar, current_chord_idx).
        
        Args:
            mmap_prefix: Prefix of the .dat files (e.g. "data/train"). 
                         Expects "{prefix}_piano.dat" and "{prefix}_chords.dat".
            split_indices: Array of integer indices defining which segments belong to this split.
            transform: Optional transform.
        """
        self.mmap_prefix = mmap_prefix
        self.split_indices = split_indices
        self.transform = transform
        
        # Determine shape from file size or pass it as argument. 
        # Here we assume we know the total count or derive it.
        # A safer way is to open in 'r' mode and read shape.
        piano_path = f"{mmap_prefix}_piano.dat"
        chords_path = f"{mmap_prefix}_chords.dat"
        
        # Open in read-only mode to get shapes
        self.piano_data = np.memmap(piano_path, dtype='float32', mode='r')
        # Reshape logic: Total bytes / bytes_per_sample (128*128)
        num_segments = self.piano_data.shape[0] // (PITCH_DIM * SAMPLES_PER_SEGMENT)
        self.piano_data = self.piano_data.reshape((num_segments, PITCH_DIM, SAMPLES_PER_SEGMENT))
        
        self.chords_data = np.memmap(chords_path, dtype='uint8', mode='r')
        self.chords_data = self.chords_data.reshape((num_segments, BARS_PER_SEGMENT))
        
    def __len__(self) -> int:
        """ Return the number of bars (not segments!) in the dataset. """
        return len(self.split_indices) * BARS_PER_SEGMENT

    def __getitem__(self, idx: int):
        """ 
        Get a sample by index.
        Returns:
            prev_tensor (Tensor): (1, 128, 16) - The previous bar (condition)
            curr_tensor (Tensor): (1, 128, 16) - The current bar (target)
            chord_tensor (Tensor): (1,) - The chord index for the current bar
        """
        # Map linear index (0 to N*8) to (segment_idx, bar_idx)
        list_idx = idx // BARS_PER_SEGMENT
        bar_idx = idx % BARS_PER_SEGMENT
        
        # Retrieve the physical index of the segment on disk
        physical_idx = self.split_indices[list_idx]
        
        # 1. Retrieve Piano Data (Current Bar)
        # Note: memmap slicing returns a new array, usually efficient
        full_segment = np.array(self.piano_data[physical_idx], dtype=np.float32)
        
        start_t = bar_idx * SAMPLES_PER_BAR
        end_t = start_t + SAMPLES_PER_BAR
        current_bar = full_segment[:, start_t : end_t]

        # 2. Retrieve Piano Data (Previous Bar)
        if bar_idx == 0:
            # Padding for first bar
            prev_bar = np.zeros((PITCH_DIM, SAMPLES_PER_BAR), dtype=np.float32)
        else:
            prev_start = (bar_idx - 1) * SAMPLES_PER_BAR
            prev_bar = full_segment[:, prev_start : prev_start + SAMPLES_PER_BAR]

        # 3. Retrieve Chord Data
        # chords_data shape is (Num_Segments, 8)
        current_chord_idx = self.chords_data[physical_idx, bar_idx]
        
        # 4. To Tensor
        curr_tensor = torch.from_numpy(current_bar).unsqueeze(0) # (1, 128, 16)
        prev_tensor = torch.from_numpy(prev_bar).unsqueeze(0)    # (1, 128, 16)
        chord_tensor = torch.tensor(current_chord_idx, dtype=torch.long) # Scalar tensor
        
        if self.transform:
            curr_tensor = self.transform(curr_tensor)
            prev_tensor = self.transform(prev_tensor)

        return prev_tensor, curr_tensor, chord_tensor
        