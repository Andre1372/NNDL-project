

# For plotting
import matplotlib.pyplot as plt
import librosa.display
# For MIDI handling
import pretty_midi
# For numerical operations
import numpy as np

def display_prettymidi(pm: pretty_midi.PrettyMIDI, fs=8, title="PrettyMIDI display"):
    """
    Display the piano roll of a PrettyMIDI object.
    
    Args:
        pm: A PrettyMIDI object
        fs: Sampling frequency (frames per second)
        title: Title of the plot
    """
    plt.figure(figsize=(14, 5))

    pianoroll = pm.get_piano_roll(fs=fs)

    if pianoroll.size == 0 or pianoroll.sum() == 0: # Handle empty/silent pianoroll case
        # Ensure at least 1 frame to avoid librosa crash
        n_frames = max(1, int(pm.get_end_time() * fs))
        pianoroll = np.zeros((128, n_frames))

    # Visualization with librosa
    librosa.display.specshow(
        pianoroll,
        sr=fs,           
        hop_length=1,    
        x_axis='time', 
        y_axis='cqt_note'
    )

    plt.title(title)
    plt.ylabel("Pitch (Octave)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Velocity")
    plt.tight_layout()
    plt.show()