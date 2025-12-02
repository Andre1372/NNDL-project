
# Base Dataset class from PyTorch
from torch.utils.data import Dataset
# For transformations on data
from typing import Optional, Callable
# For file operations
from pathlib import Path
# For midi processing
import pretty_midi
# To save sparse matrices
import numpy as np
import scipy.sparse as sparse

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

    def __len__(self) -> int:
        """ Return the number of samples in the dataset. """
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        """ Get a sample by index. """
        path = self.file_paths[idx]
        
        sparse_matrix = sparse.load_npz(path)
        
        dense_matrix = sparse_matrix.todense()

        if self.transform:
            dense_matrix = self.transform(dense_matrix)

        return dense_matrix

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
        