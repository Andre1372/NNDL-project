
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
        piano_roll[piano_roll > 0] = 1
        
        # Clip pitch range
        piano_roll[:self.note_start, :] = 0
        piano_roll[self.note_end:, :] = 0
        
        return piano_roll

    def _is_segment_valid(self, piano_roll: np.ndarray) -> bool:
        """Checks if a segment is valid based on dimensions, note count, and polyphony."""
        # Check dimensions
        if piano_roll.shape != (PITCH_DIM, SAMPLES_PER_SEGMENT): return False

        # Check sparsity (Minimum number of notes)
        if np.sum(piano_roll) < self.min_notes: return False

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

                        # Save segment to disk
                        save_name = f"{midi_file_path.parent.stem}_{midi_file_path.stem}_{segment_global_idx}.npz"
                        save_path = self.output_dir / save_name
                        
                        # Save as sparse matrix to save space, will be densified during consolidation
                        sparse_matrix = sparse.csr_matrix(piano_roll.astype(np.uint8))
                        sparse.save_npz(save_path, sparse_matrix)

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
        