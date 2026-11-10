
import pretty_midi
from pathlib import Path

file = Path("generated/classical_piano_v2.mid")

midi = pretty_midi.PrettyMIDI(str(file))

notes = [
    note
    for instrument in midi.instruments
    for note in instrument.notes
]

print("\n========== MIDI DIAGNOSTICS ==========")
print("Total instruments:", len(midi.instruments))
print("Total notes:", len(notes))
print("Unique pitches:", len(set(n.pitch for n in notes)))
print("Duration:", round(midi.get_end_time(), 2), "seconds")

if notes:
    print("Lowest pitch:", min(n.pitch for n in notes))
    print("Highest pitch:", max(n.pitch for n in notes))
    print("Average note duration:", round(
        sum(n.end - n.start for n in notes) / len(notes), 3
    ))

print("======================================")
