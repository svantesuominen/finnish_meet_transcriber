import argparse
import os
import sys
import tempfile
import whisper
import ffmpeg
import warnings
import shutil

# ... imports ...

# Constants
INPUT_DIR = "input"
OUTPUT_DIR = "output"
PROCESSED_DIR = "processed"

def setup_directories():
    """Creates the necessary directories if they don't exist."""
    for directory in [INPUT_DIR, OUTPUT_DIR, PROCESSED_DIR]:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)

def get_unique_filename(directory, filename):
    """
    Returns a unique filename in the given directory by appending a counter 
    if the file already exists.
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
        
    return new_filename

def extract_audio(video_path):
    """
    Extracts audio from a video file and saves it to a temporary WAV file.
    """
    try:
        # Create a temp file for audio
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        
        print(f"Extracting audio from {video_path}...")
        
        # Use ffmpeg to extract audio (mono mix)
        (
            ffmpeg
            .input(video_path)
            .output(temp_audio.name, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
        return temp_audio.name
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode()}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return None

def extract_stereo_channels(video_path):
    """
    Extracts Left and Right audio channels into separate WAV files.
    """
    try:
        left_temp = tempfile.NamedTemporaryFile(suffix="_left.wav", delete=False)
        right_temp = tempfile.NamedTemporaryFile(suffix="_right.wav", delete=False)
        left_temp.close()
        right_temp.close()
        
        print(f"Extracting stereo channels from {video_path}...")
        
        # Extract Left
        (
            ffmpeg
            .input(video_path)
            .filter('channelsplit', channel_layout='stereo', channels='FL')
            .output(left_temp.name, acodec='pcm_s16le', ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
        
        # Extract Right
        (
            ffmpeg
            .input(video_path)
            .filter('channelsplit', channel_layout='stereo', channels='FR')
            .output(right_temp.name, acodec='pcm_s16le', ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
        
        return left_temp.name, right_temp.name
        
    except ffmpeg.Error as e:
        print(f"Error extracting stereo channels: {e.stderr.decode()}", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"An error occurred during stereo extraction: {e}", file=sys.stderr)
        return None, None

def transcribe_audio(audio_path, model_name="base", language="fi"):
    """
    Transcribes the audio file using OpenAI Whisper.
    """
    try:
        print(f"Loading Whisper model '{model_name}'...")
        model = whisper.load_model(model_name)
        
        print(f"Transcribing (Language: {language})... This might take a while.")
        result = model.transcribe(audio_path, language=language)
        
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        return None

def diarize_audio(audio_path, hf_token):
    """
    Performs speaker diarization using pyannote.audio.
    Returns a list of segments: [(start, end, speaker), ...]
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("Error: pyannote.audio is not installed. Please install it to use diarization.", file=sys.stderr)
        return None

    try:
        print("Loading Diarization pipeline (this downloads the model from Hugging Face)...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        
        # If model failed to load (e.g. bad token), it might return None
        if pipeline is None:
             print("Error: Failed to load pipeline. Check your HF_TOKEN and if you accepted the model terms.", file=sys.stderr)
             return None

        # Move to GPU if available (requires torch)
        # import torch
        # if torch.cuda.is_available():
        #     pipeline.to(torch.device("cuda"))

        print("Diarizing audio...")
        diarization = pipeline(audio_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # start, end, speaker
            segments.append((turn.start, turn.end, speaker))
            
        return segments
    except Exception as e:
        print(f"Error during diarization: {e}", file=sys.stderr)
        return None

def get_audio_duration(video_path):
    """
    Returns the duration of the media file in seconds.
    Returns 0 if detection fails.
    """
    try:
        probe = ffmpeg.probe(video_path)
        format_info = probe.get('format', {})
        return float(format_info.get('duration', 0))
    except Exception as e:
        print(f"Warning: Could not detect duration for {video_path}: {e}")
        return 0

def format_duration(seconds):
    """Formats seconds into MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"

def transcribe_audio(audio_path, model_name="base", language="fi"):
    """
    Transcribes the audio file using OpenAI Whisper.
    """
    try:
        print(f"Loading Whisper model '{model_name}'...")
        model = whisper.load_model(model_name)
        
        print(f"Transcribing (Language: {language})... Text will appear below as it's processed:")
        # verbose=True prints segments to stdout as they are generated
        result = model.transcribe(audio_path, language=language, verbose=True)
        
        # Combine segments with newlines to avoid "wall of text"
        segments = result.get('segments', [])
        formatted_text = "\n".join([seg['text'].strip() for seg in segments])
        
        return formatted_text
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Transcribe Finnish Google Meet recordings.")
    parser.add_argument("input_file", nargs='?', help="Path to a specific video file. If not provided, scans the 'input' directory.")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help=f"Directory to save the text file (default: {OUTPUT_DIR})")
    parser.add_argument("--stereo", action="store_true", help="Process Left and Right channels separately (useful for split-channel recordings)")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization (Requires pyannote.audio and HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    setup_directories()
    
    files_to_process = []
    
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: File '{args.input_file}' not found.")
            sys.exit(1)
        files_to_process.append(args.input_file)
    else:
        # Scan for media files in INPUT_DIR
        media_extensions = ('.mp4', '.mkv', '.mov', '.avi', '.webm', '.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg')
        print(f"Scanning '{INPUT_DIR}' directory for media files...")
        
        # We only check the INPUT_DIR
        if os.path.exists(INPUT_DIR):
            for file in os.listdir(INPUT_DIR):
                if file.lower().endswith(media_extensions):
                    # Store full path
                    files_to_process.append(os.path.join(INPUT_DIR, file))
        
        if not files_to_process:
            print(f"No media files found in '{INPUT_DIR}'.")
            sys.exit(0)
            
        print(f"Found {len(files_to_process)} media file(s) to process.")

    for media_file_path in files_to_process:
        print(f"\n{'='*50}")
        print(f"Processing: {os.path.basename(media_file_path)}")
        print(f"{'='*50}")

        # Duration & Estimation
        duration = get_audio_duration(media_file_path)
        if duration > 0:
            print(f"Audio Duration: {format_duration(duration)}")
            print("\n[TIP] Performance vs Quality:")
            print(" - Current Model: " + args.model)
            if args.model in ['tiny', 'base', 'small']:
                print(" - For better accuracy (especially with elderly/muffled speech), use: --model medium")
            print(" - Large models take longer but produce the best results.")
            
            print("\nEstimated time: This depends heavily on your hardware.")
            print(" - Fast CPU/GPU: ~10-20% of audio length")
            print(" - Slower CPU:   ~50-100% of audio length")
        
        transcript = ""
        cleanup_files = []

        # Determine processing mode
        process_stereo = args.stereo
        
        # Auto-detect channels if stereo is requested
        if process_stereo:
            channels = get_audio_channels(media_file_path)
            if channels > 0 and channels < 2:
                print(f"Info: File '{os.path.basename(media_file_path)}' is detected as Mono ({channels} ch). Falling back to standard transcription.")
                process_stereo = False
            elif channels >= 2:
                print(f"Info: File '{os.path.basename(media_file_path)}' detected as Stereo/Multi-channel ({channels} ch). Processing splitting...")

        if process_stereo:
            # Stereo processing
            left_path, right_path = extract_stereo_channels(media_file_path)
            if left_path and right_path:
                cleanup_files.extend([left_path, right_path])
                
                print("\n--- Transcribing LEFT Channel ---")
                left_text = transcribe_audio(left_path, model_name=args.model, language="fi")
                
                print("\n--- Transcribing RIGHT Channel ---")
                right_text = transcribe_audio(right_path, model_name=args.model, language="fi")
                
                if left_text or right_text:
                    transcript = f"--- [LEFT CHANNEL] ---\n{left_text}\n\n--- [RIGHT CHANNEL] ---\n{right_text}"
            else:
                 print(f"Skipping {media_file_path} due to stereo extraction failure.")
                 continue

        else:
            # Standard Mono processing
            temp_audio_path = extract_audio(media_file_path)
            if not temp_audio_path:
                print(f"Skipping {media_file_path} due to audio extraction failure.")
                continue
            cleanup_files.append(temp_audio_path)
            
            # Diarization Logic
            if args.diarize:
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    print("Error: HF_TOKEN environment variable not set. Cannot perform diarization.", file=sys.stderr)
                    transcript = transcribe_audio(temp_audio_path, model_name=args.model, language="fi") # Fallback
                else:
                    # 1. Transcribe to get segments (we need word-level or segment-level timestamps, 
                    #    but standard transcribe returns a big string. We need verbose output.)
                    print(f"Loading Whisper model '{args.model}'...")
                    model = whisper.load_model(args.model)
                    print(f"Transcribing (Language: fi)... Text will appear below:")
                    # We need the result object to match timestamps
                    result = model.transcribe(temp_audio_path, language="fi", verbose=True)
                    
                    # 2. Diarize
                    speakers = diarize_audio(temp_audio_path, hf_token)
                    
                    if speakers and result:
                        print("Aligning speakers with transcript...")
                        # Simple alignment strategy:
                        # Iterate through whisper segments, find which speaker was active during that time.
                        
                        final_transcript = []
                        current_speaker = None
                        
                        # Map raw speaker labels (SPEAKER_00) to friendly names (Person 1)
                        # We'll just sort them to ensure consistent numbering
                        unique_speakers = sorted(list(set(s[2] for s in speakers)))
                        speaker_map = {label: f"Person {i+1}" for i, label in enumerate(unique_speakers)}
                        
                        for segment in result['segments']:
                            start = segment['start']
                            end = segment['end']
                            text = segment['text'].strip()
                            
                            # Find dominant speaker in this window
                            speaker_overlaps = {}
                            for sp_start, sp_end, sp_label in speakers:
                                overlap_start = max(start, sp_start)
                                overlap_end = min(end, sp_end)
                                overlap_dur = max(0, overlap_end - overlap_start)
                                if overlap_dur > 0:
                                    speaker_overlaps[sp_label] = speaker_overlaps.get(sp_label, 0) + overlap_dur
                            
                            best_speaker_label = "UNKNOWN"
                            if speaker_overlaps:
                                best_speaker_label = max(speaker_overlaps, key=speaker_overlaps.get)
                            
                            friendly_name = speaker_map.get(best_speaker_label, "Unknown Speaker")
                            
                            # Validates if we should start a new line or append to previous
                            if friendly_name != current_speaker:
                                # New speaker block
                                final_transcript.append(f"\n{friendly_name}: \"{text}\"")
                                current_speaker = friendly_name
                            else:
                                # Same speaker, continue text
                                # We append to the last item
                                final_transcript[-1] = final_transcript[-1][:-1] + f" {text}\""

                        transcript = "\n".join(final_transcript).strip()
                    else:
                        print("Diarization failed or no speakers found, falling back to simple transcript.")
                        transcript = result["text"]

            else:
                # Transcribe
                transcript = transcribe_audio(temp_audio_path, model_name=args.model, language="fi")
        
        # Cleanup temp files
        for temp_file in cleanup_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
        if transcript:
            # Construct output filename
            base_name = os.path.basename(media_file_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            
            # Ensure output directory exists (args.output_dir might be different from OUTPUT_DIR if overridden)
            if not os.path.exists(args.output_dir):
                 os.makedirs(args.output_dir)

            output_path = os.path.join(args.output_dir, f"{file_name_without_ext}.txt")
            
            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript)
                
            print(f"\nTranscription complete! Saved to: {output_path}")
            
            # Move media file to processed folder ONLY if it was picked up from the input folder
            if os.path.dirname(os.path.abspath(media_file_path)) == os.path.abspath(INPUT_DIR):
                processed_filename = get_unique_filename(PROCESSED_DIR, base_name)
                dest_path = os.path.join(PROCESSED_DIR, processed_filename)
                
                try:
                    shutil.move(media_file_path, dest_path)
                    print(f"Moved media file to: {dest_path}")
                except Exception as e:
                    print(f"Warning: Failed to move media file: {e}")

        else:
            print(f"Transcription failed for {media_file_path}.")

if __name__ == "__main__":
    main()
