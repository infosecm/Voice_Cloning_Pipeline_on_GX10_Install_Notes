#!/usr/bin/env python3
"""
Voxtral TTS — Python wrapper for batch generation.
Usage:
    python voxtral_generate.py "Your text here." [options]
    python voxtral_generate.py --file script.txt [options]

Options:
    --voice     Voice name (default: fr_female)
    --output    Output WAV file (default: ~/voice_clone/audio_samples/output_voxtral_TIMESTAMP.wav)
    --file      Read text from file instead of argument
    --no-trim   Skip silence trimming
"""

import argparse
import subprocess
import os
import sys
import datetime

VOICE_CLONE_DIR = os.path.expanduser("~/voice_clone")
BINARY = os.path.join(VOICE_CLONE_DIR, "voxtral", "voxtral_tts")
MODEL_DIR = os.path.join(VOICE_CLONE_DIR, "voxtral", "voxtral-tts-model")
SAMPLES_DIR = os.path.join(VOICE_CLONE_DIR, "audio_samples")


def generate(text, voice, output_path, trim=True):
    # Add sacrificial word to avoid first-syllable cutoff
    prefixed_text = "Donc, " + text if not text.startswith("Donc,") else text

    raw_output = output_path.replace(".wav", "_raw.wav")

    cmd = [
        BINARY,
        "-d", MODEL_DIR,
        "-v", voice,
        "-o", raw_output,
        prefixed_text
    ]

    print(f"Generating: {voice} → {os.path.basename(output_path)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"Error: voxtral_tts exited with code {result.returncode}")
        sys.exit(1)

    if trim:
        trim_cmd = [
            "ffmpeg", "-y",
            "-i", raw_output,
            "-af", "silenceremove=start_periods=1:start_threshold=-50dB",
            output_path
        ]
        subprocess.run(trim_cmd, capture_output=True)
        os.remove(raw_output)
    else:
        os.rename(raw_output, output_path)

    duration_cmd = ["ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", output_path]
    try:
        dur = subprocess.check_output(duration_cmd, stderr=subprocess.DEVNULL).decode().strip()
        print(f"Output: {output_path} ({float(dur):.1f}s)")
    except Exception:
        print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Voxtral TTS wrapper")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--voice", default="neutral_female", help="Voice name (default: neutral_female)")
    parser.add_argument("--output", default=None, help="Output WAV file path")
    parser.add_argument("--file", default=None, help="Read text from file")
    parser.add_argument("--no-trim", action="store_true", help="Skip silence trimming")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        sys.exit(1)

    if args.output:
        output_path = os.path.expanduser(args.output)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(SAMPLES_DIR, f"output_voxtral_{timestamp}.wav")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generate(text, args.voice, output_path, trim=not args.no_trim)


if __name__ == "__main__":
    main()
