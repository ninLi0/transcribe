# WhisperX Transcription CLI

A simple command-line interface (CLI) for transcribing audio files using WhisperX (large-v2 model), performing forced alignment, speaker diarization, and exporting VTT subtitle files. The script reads configuration values (Hugging Face token and output directory) from a `.env` file.

---

## Features

- **WhisperX transcription** (batched) using any WhisperX model size (default: `large-v2`).
- **Forced alignment** of transcription segments to produce precise word timestamps.
- **Speaker diarization** via Hugging Face’s pre-trained pipeline.
- **Word-level speaker assignment** integrated into the final output.
- **VTT subtitle export** (karaoke-style highlighting optional).
- All configuration (HF token, default output folder) is centralized in a `.env` file.

---

## Requirements

- Python 3.8 or later  
- A CUDA-capable GPU for `--device cuda` + `compute_type float16` (optional). Otherwise runs on CPU.
- A valid **Hugging Face token** (for the diarization pipeline).

### Python Dependencies

- `torch`  
- `whisperx`  
- `python-dotenv`  
- `argparse` (built-in)  
- `ffmpeg` (system dependency, for loading MP3/MP4 audio files)

You can install Python dependencies with:
```bash
pip install --no-cache-dir -r requirements.txt
````

> **Note:**
>
> * WhisperX may require a GPU and appropriate CUDA-compatible PyTorch build for best performance.
> * Make sure `ffmpeg` (or `sox`) is installed on your system so that `whisperx.load_audio()` can load MP3, WAV, or other formats.

---

## Project Structure

```
.
├── .env
├── models/                      # (auto-created) where WhisperX downloads model weights
├── output/                      # (auto-created) where VTT files are saved
├── whisper.py   # Main CLI script
└── README.md                    # This file
```

* **`.env`**:
  Holds environment variables for `HF_TOKEN` and optional `OUTPUT_DIR`.

* **`whisper.py`**:
  The entry-point script.

  * Reads `HF_TOKEN` and `OUTPUT_DIR` from `.env`.
  * Accepts command-line arguments for `audio_file`, `--device`, `--model_size`, `--compute_type`.
  * Runs transcription → alignment → diarization → VTT export.

---

## Setup

1. **Clone or download this repository** to your local machine.

2. **Create and activate a Python virtual environment (optional but recommended):**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install required Python packages:**

   ```bash
   pip install torch whisperx python-dotenv
   ```

4. **Install `ffmpeg`** (or `sox`) on your system if you plan to load MP3/MP4 files.
   Examples:

   * **Debian/Ubuntu:**

     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```

   * **macOS (Homebrew):**

     ```bash
     brew install ffmpeg
     ```

5. **Create a `.env` file** in the project root and add the following lines:

   ```
   HF_TOKEN=your_hugging_face_token_here
   OUTPUT_DIR=./output
   ```

   * Replace `your_hugging_face_token_here` with your actual Hugging Face access token.
   * If you omit `OUTPUT_DIR`, the script will default to `./output`.

---

## Usage

```bash
python whisper.py <audio_file> [options]
```

### Positional Arguments

* `<audio_file>`
  Path to your input audio file.
  Supported formats: WAV, MP3, M4A, etc.
  Example: `./samples/interview.wav`

### Optional Arguments

* `--device {cpu,cuda}`
  Choose inference device.

  * `cpu` (default if no GPU available)
  * `cuda` (requires a CUDA-compatible GPU)

* `--model_size MODEL_SIZE`
  WhisperX model size to load. Default is `large-v2`.
  Available sizes (depending on your installation):

  * `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`

* `--compute_type {float16,int8}`
  Precision for model weights.

  * `float16` (FP16, requires GPU)
  * `int8` (quantized, CPU/GPU)

> **Example:**
>
> ```bash
> python whisper.py samples/interview.mp3 --device cuda --model_size large-v2 --compute_type float16
> ```

---

## How It Works

1. **Load `.env`**
   The script uses `python-dotenv` to load environment variables at startup.

   * `HF_TOKEN` is mandatory (for diarization).
   * `OUTPUT_DIR` is optional (defaults to `./output`).

2. **Transcription (WhisperX)**

   * Loads the specified WhisperX model to the chosen device.
   * Runs batched transcription (`batch_size=16`).
   * Produces a list of raw segments (with approximate word timestamps, if available).

3. **Forced Alignment**

   * Loads a small alignment model (CPU or GPU).
   * Aligns each segment to generate precise word-level start/end times.

4. **Speaker Diarization**

   * Uses Hugging Face’s pre-trained pipeline (requires `HF_TOKEN`).
   * Returns speaker-labeled time segments.

5. **Word-Level Speaker Assignment**

   * Maps diarization segments to each word in the aligned transcript.
   * Each word now has a `{ start, end, speaker }` structure.

6. **VTT Subtitle Export**

   * Uses WhisperX’s built-in writer to format WebVTT.
   * Saves a single `.vtt` file to `OUTPUT_DIR` with the same base name as the input audio.

---

## Example

Assuming you have:

* Audio file: `interview.mp3`
* Valid Hugging Face token in `.env`
* GPU available

```bash
# Ensure .env contains:
#   HF_TOKEN=hf_xxxyourtokenxxx
#   OUTPUT_DIR=./my_vtt_folder

python whisper.py interview.mp3 \
    --device cuda \
    --model_size large-v2 \
    --compute_type float16
```

After successful completion, you should see:

```
Loading WhisperX model 'large-v2' (float16) on cuda...
Loading audio from 'interview.mp3'...
Running transcription with WhisperX...
Loading alignment model...
Aligning transcription segments...
Initializing diarization pipeline...
Running speaker diarization on audio...
Assigning speaker labels to words...
Writing VTT file...
Saved .vtt subtitles to './my_vtt_folder/interview.vtt'
```

Open `./my_vtt_folder/interview.vtt` in any VTT-compatible player or editor (e.g., VLC, VS Code with a VTT plugin).

---

## Troubleshooting

* **`ffmpeg` not found**

  * Install `ffmpeg` on your system or specify a WAV input instead.

* **GPU out of memory**

  * Try `--compute_type int8` (lower precision).
  * Switch to `--device cpu` if no GPU is available.

* **Invalid or missing `HF_TOKEN`**

  * Verify that your Hugging Face token is correct and has `read` permission for the diarization model.
  * Double-check your `.env` file is in the same directory as `whisper.py`.

* **WhisperX model download errors**

  * Check network connectivity.
  * Ensure sufficient disk space for model weights.

---

## Customization

* **Change default batch size**

  * In the code, adjust `batch_size = 16` under “Running transcription with WhisperX.”

* **Write SRT instead of VTT**

  * Modify `output_format = "vtt"` to `"srt"` and adjust `options` if needed.
  * Rename “`.vtt`” extension logic at the end to `.srt`.

* **Add logging or progress bars**

  * Insert `print()` or use Python’s `logging` module around key steps.
