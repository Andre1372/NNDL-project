import pretty_midi
import numpy as np
import torch

from src.data.dataset import identify_chord, process_to_monophonic_continuous

def get_test_sequence_from_midi(midi_path, note_start=24, note_end=96):
    """
    Legge un file MIDI ed estrae battute e accordi usando le funzioni 
    ufficiali del training set, mantenendo il brano in memoria.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    
    # 1. Troviamo il primo strumento valido (no drum)
    piano_instr = next((instr for instr in pm.instruments if not instr.is_drum), None)
    if not piano_instr:
        raise ValueError("Nessuno strumento valido trovato.")
        
    # 2. Stessa formula del dataset per fs
    tempo_times, tempi = pm.get_tempo_changes()
    bpm = tempi[0] if len(tempi) > 0 else 120.0
    fs = bpm / 15.0  
    
    # 3. Estraiamo e filtriamo il piano roll
    piano_roll = piano_instr.get_piano_roll(fs=fs)
    piano_roll = (piano_roll > 0).astype(np.float32)
    piano_roll[:note_start, :] = 0
    piano_roll[note_end:, :] = 0
    
    # Adattiamo la lunghezza a multipli di 16 (una battuta)
    SAMPLES_PER_BAR = 16
    num_bars = piano_roll.shape[1] // SAMPLES_PER_BAR
    piano_roll = piano_roll[:, :num_bars * SAMPLES_PER_BAR]
    
    # 4. USO DELLA TUA FUNZIONE: Estrazione Accordi
    chords = []
    for b in range(num_bars):
        bar_roll = piano_roll[:, b*SAMPLES_PER_BAR : (b+1)*SAMPLES_PER_BAR]
        chords.append(identify_chord(bar_roll)) # <--- Chiamata diretta
        
    # 5. USO DELLA TUA FUNZIONE: Monofonia
    piano_roll_mono = process_to_monophonic_continuous(piano_roll) # <--- Chiamata diretta
    
    # 6. Splittiamo in una lista di battute
    bars = [piano_roll_mono[:, b*SAMPLES_PER_BAR : (b+1)*SAMPLES_PER_BAR] for b in range(num_bars)]
        
    return np.array(bars), np.array(chords)