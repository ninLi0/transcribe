# transcribe_whisperx_cli.py

import os
import whisperx
import torch
import gc
import argparse

# Load environment variables from a .env file (requires python-dotenv to be installed)
# You can install it via: pip install python-dotenv
from dotenv import load_dotenv

def main():
    """
    Command-line interface for transcribing an audio file with WhisperX (large-v2 model),
    performing alignment, speaker diarization, and exporting a VTT subtitle file.
    The Hugging Face token (HF_TOKEN) and output directory (OUTPUT_DIR) are read from a .env file.
    """

    # 1. Load .env into environment
    load_dotenv()  # searches for a .env file in the current working directory and loads variables

    # 2. Argument parsing for audio file and optional device/model_size/compute_type
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using WhisperX (large-v2), align, diarize, and save VTT subtitles."
    )
    parser.add_argument(
        "audio_file",
        help="Path to the input audio file (e.g., /path/to/audio.mp3)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run inference on: 'cuda' or 'cpu' (default: cuda)"
    )
    parser.add_argument(
        "--model_size",
        default="large-v2",
        help="WhisperX model size to load (default: large-v2)"
    )
    parser.add_argument(
        "--compute_type",
        choices=["float16", "int8"],
        default="float16",
        help="Precision type for the Whisper model: float16 or int8 (default: float16)"
    )
    args = parser.parse_args()

    audio_file = args.audio_file
    device = args.device
    model_size = args.model_size
    compute_type = args.compute_type

    # 3. Read HF_TOKEN and OUTPUT_DIR from environment variables
    hf_token = os.getenv("HF_TOKEN")
    output_dir = os.getenv("OUTPUT_DIR", "./output")  # default to ./output if not set

    if not hf_token:
        print("Error: HF_TOKEN not set in environment. Please add HF_TOKEN to your .env file.")
        return

    # 4. Validate audio file path
    if not os.path.isfile(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        return

    # 5. Prepare directories
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 6. Load Whisper model and transcribe
    print(f"Loading WhisperX model '{model_size}' ({compute_type}) on {device}...")
    model = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        download_root=model_dir,
        language="en"  # force English
    )

    print(f"Loading audio from '{audio_file}'...")
    audio = whisperx.load_audio(audio_file)

    print("Running transcription with WhisperX...")
    batch_size = 16
    result = model.transcribe(audio, batch_size=batch_size)
    # result["segments"] contains raw segments before alignment

    # Optionally free VRAM if using CUDA
    # gc.collect()
    # torch.cuda.empty_cache()
    # del model

    # 7. Load alignment model and force-align segments
    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
    )

    print("Aligning transcription segments...")
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )
    # result["segments"] now contains aligned timestamps

    # Optionally free VRAM for alignment model
    # gc.collect()
    # torch.cuda.empty_cache()
    # del model_a

    # 8. Speaker diarization
    from whisperx.diarize import DiarizationPipeline

    print("Initializing diarization pipeline...")
    diarize_model = DiarizationPipeline(
        use_auth_token=hf_token,
        device=device,
    )

    print("Running speaker diarization on audio...")
    diarize_segments = diarize_model(audio)
    # diarize_segments contains time-stamped speaker clusters

    # 9. Assign speaker labels to each word
    print("Assigning speaker labels to words...")
    result = whisperx.assign_word_speakers(diarize_segments, result)
    # Now result["segments"] includes speaker labels within words

    # 10. Prepare for VTT output
    from whisperx.utils import get_writer

    output_format = "vtt"
    options = {
        "highlight_words": False,   # no karaoke-style highlighting
        "max_line_width": None,     # no forced line wrapping
        "max_line_count": None      # no forced line-count wrapping
    }

    # Ensure language key exists
    result["language"] = result.get("language", "en")

    print("Writing VTT file...")
    writer = get_writer(output_format, output_dir)
    writer(result, audio_file, options)

    # Construct saved VTT path
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    vtt_path = os.path.join(output_dir, f"{base_name}.vtt")

    print(f"Saved .vtt subtitles to '{vtt_path}'")

if __name__ == "__main__":
    main()
