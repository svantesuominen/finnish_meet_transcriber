# Finnish Meet Transcriber

A Python tool for automatically transcribing Finnish Google Meet recordings (and other media files) using OpenAI's Whisper model.

## Features

- **Automatic Transcription**: Uses OpenAI's `whisper` model to transcribe audio to text.
- **Finnish Language Support**: Specifically configured for Finnish (`language="fi"`).
- **Batch Processing**: Automatically scans the `input/` folder for media files.
- **Workflow Automation**:
  - Moves processed videos to `processed/`.
  - Saves transcripts to `output/`.
- **Media Support**: Supports `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.mp3`, `.wav`, `.m4a`, `.flac`, `.aac`, `.ogg`.

## Usage

1. **Install Dependencies**:
   Ensure you have Python installed and the required packages:
   ```bash
   pip install openai-whisper ffmpeg-python
   ```
   *Note: You also need `ffmpeg` installed on your system.*

2. **Prepare Directories**:
   The script will automatically create `input/`, `output/`, and `processed/` directories if they don't exist.

3. **Run**:
   Place your video or audio files in the `input/` directory and run:
   ```bash
   python transcriber.py
   ```

   The script will:
   - Find media files in `input/`.
   - Extract audio and transcribe it.
   - Save the transcript to `output/<filename>.txt`.
   - Move the original media file to `processed/`.

## Credits

This project was co-created by **Svante Suominen** and **Antigravity** (Google DeepMind).
