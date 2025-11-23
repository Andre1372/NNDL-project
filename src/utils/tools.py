
# for file operations
from pathlib import Path
# To work with sparse matrices
import numpy as np
import scipy.sparse as sparse
# For midi processing
import pretty_midi

def npz_to_midi_file(npz_path: Path, note_start, fs=100):
    """
    Carica un file npz, ricostruisce il piano roll completo e salva un MIDI.
    Args:
        npz_path: Percorso al file .npz contenente la matrice sparsa del piano roll
        note_start: La nota MIDI corrispondente alla riga 0 della matrice salvata
        fs: Frequenza di campionamento (frame al secondo)
    Returns:
        pm: Un oggetto PrettyMIDI rappresentante il piano roll
    """
    # Load sparse matrix from npz file
    sparse_matrix = sparse.load_npz(npz_path)
    
    # Convert to dense numpy array
    piano_roll_slice = sparse_matrix.toarray()
    
    current_height = piano_roll_slice.shape[0] 

    # Reconstruct the full matrix 128xTime (Standard MIDI size)   
    if current_height < 128:
        pad_bottom = note_start
        pad_top = 128 - (pad_bottom + current_height)
        
        full_piano_roll = np.pad(
            piano_roll_slice,
            pad_width=((pad_bottom, pad_top), (0, 0)), 
            mode='constant',
            constant_values=0
        )
    else:
        full_piano_roll = piano_roll_slice

    # Convert to PrettyMIDI object
    pm = piano_roll_to_pretty_midi(full_piano_roll, fs=fs, program=0)
    
    return pm

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    """
    Convert a Piano Roll array into a PrettyMidi object.
    Args:
        piano_roll: 2D numpy array (Pitches x Time)
        fs: Sampling frequency (frames per second)
        program: MIDI program number for the instrument
    Returns:
        pm: A PrettyMidi object representing the piano roll
    """
    notes, _ = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # Pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # Use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # Keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # Use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm