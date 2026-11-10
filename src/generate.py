
import torch
import numpy as np
import pretty_midi

from pathlib import Path
from model import MusicLSTM

# ---------------- CONFIGURATION ----------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path("models/best_music_model.pth")
DATA_PATH = Path("preprocessed/training_sequences.npz")
OUTPUT_DIR = Path("generated")

OUTPUT_DIR.mkdir(exist_ok=True)

NUM_NOTES = 256
TEMPERATURE = 0.8

print("=" * 45)
print("       AI MUSIC GENERATION")
print("=" * 45)

print("Device:", DEVICE)

# ---------------- LOAD MODEL ----------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = MusicLSTM(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Trained model loaded successfully!")

# ---------------- LOAD SEED SEQUENCE ----------------

data = np.load(DATA_PATH)

X = data["X"]

random_index = np.random.randint(len(X))

seed = X[random_index].tolist()

generated_notes = seed.copy()

print("Initial sequence length:", len(seed))

# ---------------- GENERATE NOTES ----------------

print("\nGenerating new musical notes...")

with torch.no_grad():

    for _ in range(NUM_NOTES):

        input_tensor = torch.tensor(
            [generated_notes[-32:]],
            dtype=torch.long,
            device=DEVICE
        )

        prediction = model(input_tensor)

        # Restrict output to standard piano range
        logits = prediction[0].clone()
        logits[:21] = float("-inf")
        logits[109:] = float("-inf")

        # Temperature controls randomness
        probabilities = torch.softmax(
            logits / TEMPERATURE,
            dim=-1
        )

        next_note = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        generated_notes.append(next_note)

print("Generated notes:", NUM_NOTES)

# ---------------- CONVERT TO MIDI ----------------

print("\nConverting notes to MIDI...")

midi = pretty_midi.PrettyMIDI(
    initial_tempo=110
)

piano = pretty_midi.Instrument(
    program=pretty_midi.instrument_name_to_program(
        "Acoustic Grand Piano"
    )
)

# Use the generated notes only
new_notes = generated_notes[-NUM_NOTES:]

current_time = 0.0
note_interval = 0.30
note_duration = 0.25

for pitch in new_notes:

    note = pretty_midi.Note(
        velocity=90,
        pitch=int(pitch),
        start=current_time,
        end=current_time + note_duration
    )

    piano.notes.append(note)

    current_time += note_interval

midi.instruments.append(piano)

# ---------------- SAVE MUSIC ----------------

output_file = OUTPUT_DIR / "ai_generated_music.mid"

midi.write(str(output_file))

print("\n" + "=" * 45)
print("       MUSIC GENERATION COMPLETE")
print("=" * 45)

print("Generated notes:", len(new_notes))
print("Output file:", output_file)
print("Duration:", round(current_time, 2), "seconds")

print("\nYour AI-generated music is ready!")
