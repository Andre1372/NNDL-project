

# For plotting
import matplotlib.pyplot as plt
import librosa.display
# For MIDI handling
import pretty_midi

def display_prettymidi(pm: pretty_midi.PrettyMIDI, fs: int, pitch_range = (0,127), title="PrettyMIDI display"):
    """
    Display the piano roll of a PrettyMIDI object.
    
    Args:
        pm: A PrettyMIDI object
        fs: Sampling frequency (frames per second)
        pitch_range: Tuple (min_pitch, max_pitch) to display specific pitch range.
        title: Title of the plot
    """
    plt.figure(figsize=(14, 5))

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