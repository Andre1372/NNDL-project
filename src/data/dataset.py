
from torch.utils.data import Dataset
from typing import Optional, Callable

# For file operations
import os
from pathlib import Path
# For parallel processing
from tqdm.contrib.concurrent import process_map 
# For midi processing
import pretty_midi
# To save sparse matrices
import numpy as np
import scipy.sparse as sparse

REPO_ROOT = Path(__file__).parents[2].resolve()

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

    def __init__(self, file_paths,transform: Optional[Callable] = None):
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
    def __init__(self, fs: int, note_start: int, note_end: int, output_dir: Path, skipping: bool):
        """
        Initialize the MIDI preprocessor.
        Args:
            fs: Sampling frequency (frames per second)
            note_start: The starting MIDI note number (inclusive)
            note_end: The ending MIDI note number (exclusive)
            output_dir: Directory to save processed files
            skipping: Whether to skip already processed files
        """
        self.fs = fs
        self.note_start = note_start
        self.note_end = note_end
        self.output_dir = output_dir
        self.skipping = skipping

    # To let this class be called like a function
    def __call__(self, midi_file_path: Path):
        output_filename = f"{midi_file_path.stem}_sparse.npz"
        save_path = self.output_dir / output_filename

        # Skip if already processed
        if self.skipping and save_path.exists():
            return "SKIPPED"

        try:
            # Extract piano tracks
            pm = pretty_midi.PrettyMIDI(str(midi_file_path))
            piano_instruments = [instr for instr in pm.instruments if instr.program in range(8) and not instr.is_drum]
            if piano_instruments == []:
                return  None # Skip files with no piano instruments

            pm_piano = pretty_midi.PrettyMIDI()
            for piano_instr in piano_instruments: pm_piano.instruments.append(piano_instr)
            
            # Store the piano roll of the piano midi
            piano_roll = pm_piano.get_piano_roll(fs=self.fs)[self.note_start:self.note_end, :]
            
            # Save an efficient sparse representation
            sparse_piano_roll = sparse.csr_matrix(piano_roll)
            sparse.save_npz(save_path, sparse_piano_roll)

            return "PROCESSED"
        except Exception as e:
            return f"ERROR: {midi_file_path.name}: {e}"