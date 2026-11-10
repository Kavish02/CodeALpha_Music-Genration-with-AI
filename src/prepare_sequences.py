
import pretty_midi
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# Configuration
DATASET_PATH = Path("dataset")
OUTPUT_PATH = Path("preprocessed")

SEQUENCE_LENGTH = 32
STRIDE = 4
MAX_SEQUENCES = 100_000

OUTPUT_PATH.mkdir(exist_ok=True)

# Find MIDI files
midi_files = list(DATASET_PATH.rglob("*.mid"))
midi_files += list(DATASET_PATH.rglob("*.midi"))

# Shuffle files reproducibly
random.Random(42).shuffle(midi_files)

# Allocate memory for training data
X = np.empty(
    (MAX_SEQUENCES, SEQUENCE_LENGTH),
    dtype=np.int16
)

y = np.empty(MAX_SEQUENCES, dtype=np.int64)

count = 0
successful_files = 0

print("=" * 45)
print("       PREPARING LSTM TRAINING DATA")
print("=" * 45)

print("MIDI files found:", len(midi_files))

for file in tqdm(midi_files, desc="Processing MIDI"):

    if count >= MAX_SEQUENCES:
        break

    try:
        midi = pretty_midi.PrettyMIDI(str(file))

        # Extract notes from this performance only
        notes = []

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            notes.extend(instrument.notes)

        # Sort notes by starting time
        notes.sort(key=lambda note: note.start)

        pitches = [note.pitch for note in notes]

        if len(pitches) < SEQUENCE_LENGTH + 1:
            continue

        successful_files += 1

        # Create input and target sequences
        for i in range(
            0,
            len(pitches) - SEQUENCE_LENGTH,
            STRIDE
        ):

            if count >= MAX_SEQUENCES:
                break

            X[count] = pitches[i:i + SEQUENCE_LENGTH]
            y[count] = pitches[i + SEQUENCE_LENGTH]

            count += 1

    except Exception as e:
        print(f"\nError processing {file.name}: {e}")

# Keep only generated samples
X = X[:count]
y = y[:count]

# Save training dataset
output_file = OUTPUT_PATH / "training_sequences.npz"

np.savez_compressed(
    output_file,
    X=X,
    y=y
)

print("\n" + "=" * 45)
print("       SEQUENCE PREPARATION COMPLETE")
print("=" * 45)

print("Successfully processed files:", successful_files)
print("Training samples:", len(X))
print("Input shape:", X.shape)
print("Target shape:", y.shape)
print("Unique pitches:", len(np.unique(X)))
print("Saved to:", output_file)
