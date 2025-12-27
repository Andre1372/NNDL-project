
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
    
class PianoMmapDataset(Dataset):
    def __init__(self, mmap_path: str, split_indices: np.ndarray, shape: tuple, transform=None):
        """
        Initialize the dataset.
        Args:
            mmap_path: Path to file .dat
            split_indices: Array of indices (integers) corresponding to the segments 
                           assigned to this split (e.g., only training segments).
            shape: Tuple representing the shape of the entire dataset on disk (e.g.; num_segments, 128, 128)
            transform: Optional transformations (e.g., ToTensor)
        """
        self.mmap_path = mmap_path
        self.split_indices = split_indices  # This is the "access mask"
        self.shape = shape
        self.transform = transform
        self.data = np.memmap(mmap_path, dtype='uint8', mode='r', shape=self.shape)
        
        # Constants
        self.bars_per_segment = 8
        self.steps_per_bar = 16
        self.pitch_dim = 128

    def __len__(self) -> int:
        """ Return the number of bars in the dataset. """
        return len(self.split_indices) * self.bars_per_segment

    def __getitem__(self, idx: int):
        """ Get a sample by index. """
        # The dataloader idx goes from 0 to len(self).
        list_idx = idx // self.bars_per_segment  # Index in the split_indices list
        bar_idx = idx % self.bars_per_segment    # Which bar (0-7)
        
        # Now retrieve the TRUE index on disk
        physical_segment_idx = self.split_indices[list_idx]
        
        # Data Access (Memory Mapped)
        full_segment = np.array(self.data[physical_segment_idx], dtype=np.float32)
        
        # Slicing Current Bar
        start_t = bar_idx * self.steps_per_bar
        current_bar = full_segment[:, start_t : start_t+self.steps_per_bar]

        if bar_idx == 0:
            # Empty padding for the first bar
            prev_bar = np.zeros((self.pitch_dim, self.steps_per_bar), dtype=np.float32)
        else:
            # Slicing Previous Bar
            prev_start = (bar_idx - 1) * self.steps_per_bar
            prev_bar = full_segment[:, prev_start : prev_start+self.steps_per_bar]

        # ToTensor
        curr_tensor = torch.from_numpy(current_bar).unsqueeze(0)
        prev_tensor = torch.from_numpy(prev_bar).unsqueeze(0)
        
        if self.transform:
            curr_tensor = self.transform(curr_tensor)
            prev_tensor = self.transform(prev_tensor)

        return prev_tensor, curr_tensor

class MidiPreprocessor:
    def __init__(self, select_instruments: list, note_start: int, note_end: int, output_dir: Path, min_notes: int = 5, min_polyphony: float = 1.0):
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
        self.min_notes = min_notes
        self.min_polyphony = min_polyphony

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
                        
                        note_count = np.sum(window)

                        notes_per_step = np.sum(window, axis=0) 
                        active_steps = notes_per_step[notes_per_step > 0] # Consideriamo solo i momenti in cui si suona
                        avg_polyphony = np.mean(active_steps) if len(active_steps) > 0 else 0

                        # --- APPLICAZIONE EURISTICHE ---
                        if note_count < self.min_notes:
                            continue  # Scarta segmenti troppo "sparsi" o quasi vuoti

                        if avg_polyphony < self.min_polyphony:
                            continue  # Scarta segmenti che sembrano strumenti monofonici (es. flauti o bassi)

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