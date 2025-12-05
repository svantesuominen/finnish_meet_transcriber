# Finnish Meet Transcriber - Walkthrough

This document outlines the features and usage of the **Finnish Meet Transcriber** tool.

## Key Features
- **Auto-Transcription**: Uses OpenAI's Whisper model to transcribe Finnish audio/video.
- **Organization**: Automatically manages `input`, `output`, and `processed` folders.
- **Smart Formatting**:
    - **Standard**: Continuous text for single speakers.
    - **Diarization**: "Person 1 / Person 2" dialogue format for multi-speaker meetings.
    - **Stereo**: Split detection for files with separate Left/Right channels.
- **Progress Feedback**: Shows audio duration, time estimates, and real-time transcription logs.

## Quick Start Guide

1.  **Drop your files** (mp4, mp3, m4a, etc.) into the `input/` folder.
2.  **Run the script**:

    ### 1. Basic Transcription (Fast)
    Good for single speakers or quick drafts.
    ```bash
    python transcriber.py
    ```

    ### 2. High Quality (Recommended for Phone/Elderly)
    Uses the "Medium" brain for better accuracy.
    ```bash
    python transcriber.py --model medium
    ```

    ### 3. Speaker Separation (Diarization)
    **Use this for meetings.** Requires Hugging Face token.
    Separates "Person 1" and "Person 2".
    ```bash
    python transcriber.py --diarize --model medium
    ```
    *Output:*
    ```text
    Person 1: "Mitä kuuluu?"
    Person 2: "Hyvää kiitos!"
    ```

    ### 4. Stereo Files
    If you recorded separate channels (Left/Right).
    ```bash
    python transcriber.py --stereo
    ```

## Tips
- **Nonsense words?** The AI might be hallucinating due to low-quality audio. **Use `--model medium` or `--model large`**.
- **Installation Issues?** If `--diarize` fails, ensure you have:
  1. `pyannote.audio` installed.
  2. `HF_TOKEN` environment variable set.
