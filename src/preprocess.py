
import pretty_midi
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

DATASET_PATH = Path("dataset")
OUTPUT_PATH = Path("preprocessed")

OUTPUT_PATH.mkdir(exist_ok=True)

# Find MIDI files
midi_files = list(DATASET_PATH.rglob("*.mid"))
midi_files += list(DATASET_PATH.rglob("*.midi"))

all_notes = []

print("Starting MIDI preprocessing...")
print("Total files:", len(midi_files))

for file in tqdm(midi_files):

    try:
        midi = pretty_midi.PrettyMIDI(str(file))

        for instrument in midi.instruments:

            if instrument.is_drum:
                continue

            for note in instrument.notes:

                all_notes.append({
                    "pitch": note.pitch,
                    "start": note.start,
                    "end": note.end,
                    "duration": note.end - note.start,
                    "velocity": note.velocity
                })

    except Exception as e:
        print(f"\nSkipping {file.name}: {e}")

print("\nTotal extracted notes:", len(all_notes))

# Save extracted notes
output_file = OUTPUT_PATH / "all_notes.pkl"

with open(output_file, "wb") as f:
    pickle.dump(all_notes, f)

print("Saved:", output_file)

# Display sample notes
print("\nFirst 10 extracted notes:")

for note in all_notes[:10]:
    print(note)

print("\nMIDI preprocessing completed!")
