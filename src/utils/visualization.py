import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def display_chromagram(chromagram, title="Chromagram", fs=100):
    """
    Display a chromagram using matplotlib.
    
    Args:
        chromagram: 2D numpy array representing the chromagram
        title: Title of the plot
        fs: Sampling frequency (frames per second)
    """
    if chromagram.shape[0] != 12 and chromagram.shape[1] == 12:
        chromagram = chromagram.T
    
    if chromagram.shape[0] != 12:
        print("Warning: The provided chromagram does not have 12 pitch classes.")
        return
    
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
