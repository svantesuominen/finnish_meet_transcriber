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

   **Standard Transcription:**
   ```bash
   python transcriber.py
   ```

   **Stereo Mode (Left/Right Channel Split):**
   Useful if you have recorded speakers on separate channels.
   ```bash
   python transcriber.py --stereo
   ```

   **Speaker Diarization (AI Speaker Detection):**
   Requires `pyannote.audio` and a Hugging Face Token.
   1. Install pyannote: `pip install pyannote.audio` (and `torch` with CUDA if you have a GPU)
   2. Set your token: `export HF_TOKEN="your_token_here"`
   3. Run:
   ```bash
   python transcriber.py --diarize
   ```

   The script will:
   - Find media files in `input/`.
   - Extract audio and transcribe it.
   - Save the transcript to `output/<filename>.txt`.
   - Move the original media file to `processed/`.

## Improving Accuracy (Important!)
If the results contain nonsense words or poor transcription (e.g. for elderly speakers or phone audio), **use a larger model**.
The default is `base` (fast but less accurate).

**Recommended Command for Quality:**
```bash
python transcriber.py --model medium
```
or for the absolute best results (runs slower):
```bash
python transcriber.py --model large
```

## Output Format
- **Standard**: Continuous text block.
- **Diarization (`--diarize`)**: Readable dialogue format:
  ```text
  Person 1: "Hello."
  Person 2: "Hi there."
  ```

## Credits

This project was co-created by **Svante Suominen** and **Antigravity** (Google DeepMind).
