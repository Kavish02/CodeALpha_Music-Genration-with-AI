# 🎵 AI Music Generation Using Deep Learning

An AI-powered music generation project that uses **Long Short-Term Memory (LSTM)** neural networks to learn musical patterns from MIDI datasets and generate new piano compositions.

The project uses the MAESTRO MIDI dataset to train a deep learning model capable of learning note sequences and generating original MIDI music.

## 🚀 Project Overview

The objective of this project is to develop an AI music generator using deep learning techniques.

The model learns musical patterns from existing MIDI files and uses the trained LSTM network to predict subsequent musical notes and generate new compositions.

## ✨ Features

- MIDI dataset collection and preprocessing.
- Automatic extraction of musical notes.
- Deep learning using LSTM neural networks.
- GPU acceleration using NVIDIA CUDA.
- Model training and validation.
- AI-generated MIDI music sequences.
- MIDI file export for playback.
- Training loss visualization.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Programming language |
| PyTorch | Deep learning framework |
| LSTM | Music sequence learning |
| NumPy | Numerical operations |
| PrettyMIDI | MIDI processing |
| Matplotlib | Training visualization |
| MAESTRO v3.0 | MIDI dataset |
| CUDA | GPU acceleration |
| VS Code | Development environment |

## 💻 System Requirements

- Windows 10/11
- Python 3.11
- VS Code
- NVIDIA GPU (optional)
- CUDA-compatible PyTorch installation
- Minimum 8 GB RAM recommended

## 📂 Project Structure

```text
CodeAlpha_Music Generation with AI/
│
├── dataset/
│   └── MAESTRO MIDI dataset
│
├── generated/
│   └── ai_generated_music.mid
│
├── models/
│   ├── best_music_model.pth
│   └── training_loss.png
│
├── preprocessed/
│   ├── all_notes.pkl
│   └── training_sequences.npz
│
├── src/
│   ├── check_midi.py
│   ├── generate.py
│   ├── model.py
│   ├── prepare_sequences.py
│   ├── preprocess.py
│   └── train.py
│
├── venv/
├── gpu_test.py
├── README.md
└── requirements.txt
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "CodeAlpha_Music Generation with AI"
```

### 2. Create a virtual environment

```bash
py -3.11 -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make sure PyTorch is installed with a CUDA-compatible build if you want GPU acceleration.

## 📊 Dataset

This project uses the **MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization)** dataset.

Dataset source:

https://magenta.tensorflow.org/datasets/maestro

The project uses the MIDI-only download.

Extract the downloaded MIDI files inside the `dataset/` directory.

The preprocessing scripts search recursively for MIDI files.

## 🧠 Model Architecture

The project uses an LSTM-based deep learning architecture.

The model consists of:

1. Embedding layer – represents musical notes.
2. LSTM layers – learn sequential musical patterns.
3. Fully connected layer – predicts the next musical note.

The model is trained on preprocessed MIDI note sequences.

## 🔄 Project Workflow

### Step 1: Dataset preprocessing

```bash
python src/preprocess.py
```

Extracts musical notes from MIDI files and stores the processed data.

### Step 2: Sequence preparation

```bash
python src/prepare_sequences.py
```

Converts extracted notes into sequences suitable for LSTM training.

### Step 3: Model training

```bash
python src/train.py
```

Trains the LSTM model using the prepared dataset.

The trained model and training visualization are saved in the `models/` directory.

### Step 4: Music generation

```bash
python src/generate.py
```

Uses the trained model to generate new musical sequences and export them as MIDI.

### Step 5: MIDI diagnostics

```bash
python src/check_midi.py
```

Checks the generated MIDI file, including:

- Total instruments
- Total notes
- Unique pitches
- Music duration
- Pitch range
- Average note duration

## 🎹 Generated Output

The generated MIDI file is saved at:

```text
generated/ai_generated_music.mid
```

You can open the generated MIDI file using a MIDI editor or compatible music software.

## 🎮 GPU Support

The project supports NVIDIA GPU acceleration through PyTorch CUDA.

To verify GPU availability:

```bash
python gpu_test.py
```

You can also check CUDA directly:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is available, PyTorch can use the NVIDIA GPU for model training.

## 📈 Results

The project successfully demonstrates:

- MIDI dataset preprocessing.
- Musical sequence extraction.
- LSTM model development.
- Deep learning model training.
- AI-based MIDI music generation.
- MIDI playback and analysis.

The generated compositions are experimental and can be further improved through better musical representations, training data and model architectures.

## 🔮 Future Improvements

- Generate more natural musical compositions.
- Improve melody and harmony generation.
- Introduce chord progression learning.
- Learn musical rhythm and note duration.
- Support multiple musical genres.
- Develop a web-based music generation interface.
- Export generated compositions as WAV audio.

## 👨‍💻 Author

**Kavish Vijan**

B.Tech – Computer Science and Engineering  
Specialization: Artificial Intelligence and Data Science

DY Patil University, Pune

## 📜 License

This project is developed for educational and experimental purposes.

Check the MAESTRO dataset's license and attribution requirements before redistributing its contents or derived assets