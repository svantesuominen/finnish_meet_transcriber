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
        
        # Use ffmpeg to extract audio
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

def main():
    parser = argparse.ArgumentParser(description="Transcribe Finnish Google Meet recordings.")
    parser.add_argument("input_file", nargs='?', help="Path to a specific video file. If not provided, scans the 'input' directory.")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size")
    # Removed --output_dir argument as we now enforce the structure, but could keep it as an override if needed. 
    # For now, let's keep it simple and stick to the requested structure, but respecting if user wants to override default.
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help=f"Directory to save the text file (default: {OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    setup_directories()
    
    files_to_process = []
    
    # Logic: 
    # 1. If input_file arg is provided -> Process that specific file (in place or wherever it is). 
    #    NOTE: The user request implies searching in ./input. If a specific file is given, we should probably just process it.
    #    However, the request specifically asked "make it so that it searches for the videos in ./input folder".
    #    So default behavior updates to search ./input.
    
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
        print(f"\nProcessing: {media_file_path}")
        
        # Extract audio
        temp_audio_path = extract_audio(media_file_path)
        if not temp_audio_path:
            print(f"Skipping {media_file_path} due to audio extraction failure.")
            continue
            
        # Transcribe
        transcript = transcribe_audio(temp_audio_path, model_name=args.model, language="fi")
        
        # Cleanup temp file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
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
                
            print(f"Transcription complete! Saved to: {output_path}")
            
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
