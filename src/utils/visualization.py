

# For plotting
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
# For MIDI handling
import pretty_midi

def display_chromagram(chromagram, title="Chromagram", fs=100):
    """
    Display a chromagram using matplotlib.
    
    Args:
        chromagram: 2D numpy array representing the chromagram
        title: Title of the plot
        fs: Sampling frequency (frames per second)
    """
    if chromagram.ndim != 2 or (chromagram.shape[0] != 12 and chromagram.shape[1] != 12):
        print(f"Warning: The provided chromagram has not a valid shape: {chromagram.shape}")
        return

    # Orientation check: if there are 12 columns and more than 12 rows,
    # we assume it is (Time, Pitch Classes). If the user passes (Pitch Classes, Time), we transpose.    
    if chromagram.shape[0] != 12 and chromagram.shape[1] == 12:
        chromagram = chromagram.T
    
    plt.figure(figsize=(14, 5))

    librosa.display.specshow(
        chromagram,
        sr=fs,           
        hop_length=1,    
        x_axis='time', 
        y_axis=None
    )

    note_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    plt.title(title)
    plt.yticks(ticks=np.arange(12), labels=note_labels)
    plt.colorbar(label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch Class")
    plt.tight_layout()
    plt.show()

def display_pianoroll(pianoroll, min_pitch, title="Piano Roll", fs=100):
    """
    Display a piano roll using matplotlib.
    
    Args:
        pianoroll: 2D numpy array (Pitches x Time)
        title: Title of the plot.
        fs: Sampling frequency of frames (frames per second).
        min_pitch: The MIDI number corresponding to row 0 (default 0).
    """
    if pianoroll.ndim != 2 or (pianoroll.shape[0] > 128 and pianoroll.shape[1] > 128):
        print(f"Warning: The provided pianoroll has not a valid shape: {pianoroll.shape}")
        return
    
    # Orientation check: if there are more columns than rows and the rows are few (< 128),
    # we assume it is (Pitches, Time). If the user passes (Time, Pitches), we transpose.
    if pianoroll.shape[0] > 128 and pianoroll.shape[1] <= 128:
        pianoroll = pianoroll.T
        
    plt.figure(figsize=(14, 5))

    # Visualization with librosa
    librosa.display.specshow(
        pianoroll,
        sr=fs,           
        hop_length=1,    
        x_axis='time', 
        y_axis='cqt_note',
        fmin=pretty_midi.note_number_to_hz(min_pitch)
    )

    plt.title(title)
    plt.ylabel("Pitch (Octave)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Velocity")
    plt.tight_layout()
    plt.show()

def display_prettymidi(pm: pretty_midi.PrettyMIDI, fs: int, pitch_range = None, title="PrettyMIDI display"):
    """
    Display the piano roll or chromagram of a PrettyMIDI object.
    
    Args:
        pm: A PrettyMIDI object
        fs: Sampling frequency (frames per second)
        pitch_range: Tuple (min_pitch, max_pitch) to display specific pitch range. If None, display chromagram.
        title: Title of the plot
    """
    plt.figure(figsize=(14, 5))

    # If no pitch range is given, display chormagram
    if pitch_range is None:
        chromagram = pm.get_chroma(fs=fs)
        
        librosa.display.specshow(
            chromagram,
            sr=fs,           
            hop_length=1,    
            x_axis='time', 
            y_axis=None
        )

        note_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        plt.title(title)
        plt.yticks(ticks=np.arange(12), labels=note_labels)
        plt.colorbar(label="Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch Class")
        plt.tight_layout()
        plt.show()
    else:
        pianoroll = pm.get_piano_roll(fs=fs)[pitch_range[0]:pitch_range[1], :]

        # Visualization with librosa
        librosa.display.specshow(
            pianoroll,
            sr=fs,           
            hop_length=1,    
            x_axis='time', 
            y_axis='cqt_note',
            fmin=pretty_midi.note_number_to_hz(pitch_range[0])
        )

        plt.title(title)
        plt.ylabel("Pitch (Octave)")
        plt.xlabel("Time (s)")
        plt.colorbar(label="Velocity")
        plt.tight_layout()
        plt.show()