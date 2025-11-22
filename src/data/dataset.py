
from torch.utils.data import Dataset
from typing import Optional, Callable

# for file operations
import os
from pathlib import Path
# for parallel processing
from tqdm.contrib.concurrent import process_map 
# for midi processing
import pretty_midi
# to save sparse matrices
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
    
def midi_to_matrix(midi_file_path: Path) -> np.ndarray:
    """
    Convert a MIDI file to a piano chromagram matrix.

    Args:
        midi_file_path: Path to the MIDI file
    
    Returns:
        chromagram: 2D numpy array representing the piano chromagram
    """    
    output_filename = f"{midi_file_path.stem}_sparse.npz"
    save_path = REPO_ROOT / 'data' / 'processed_midi_matrices' / output_filename

    if save_path.exists():
        return None # Saltiamo se già fatto (utile se si interrompe e riprende)
    
    try:
        # extract piano tracks
        pm = pretty_midi.PrettyMIDI(str(midi_file_path))
        piano_instruments = [instr for instr in pm.instruments if instr.program in range(8) and not instr.is_drum]
        if piano_instruments == []:
            return  None # Skip files with no piano instruments

        pm_piano = pretty_midi.PrettyMIDI()
        for piano_instr in piano_instruments: pm_piano.instruments.append(piano_instr)

        # store the chromagram of the piano midi
        chromagram = pm_piano.get_chroma(fs=100)

        # Convert to sparse format (CSR or CSC are common)
        sparse_chromagram = sparse.csr_matrix(chromagram)
        # Save using scipy
        sparse.save_npz(save_path, sparse_chromagram)
        
        return None
    except Exception as e:
        print(f"Error processing {midi_file_path.name}: {e}")
