from __future__ import annotations
from pathlib import Path
import subprocess

#repo-relative paths
RAW_DIR = Path("data/raw")
WAV_DIR = Path("data/wav")

#classes i.e. labels
LABELS = ["baroque", "classical", "romantic"]
DEFAULT_SR = 22050
EX_PATH = r"C:\Users\alice\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"

def convert_file(input_path : Path, output_path : Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
       EX_PATH, 
        "-y", #overwrite existing output
        "-i", str(input_path), #input file
        "-vn", #ignore vid
        "-ac", "1", #mono audio
        "-ar", str(DEFAULT_SR), #sample rate 22050 Hz for default
        str(output_path) #output wav path
    ]

    print(f"Converting: {input_path} -> {output_path}")
    subprocess.run(command, check = True)

def main() -> None:
    total = 0

    for label in LABELS:
        input_folder = RAW_DIR/label
        output_folder = WAV_DIR/label

        if not input_folder.exists():
            print(f"Skipping missing folder: {input_folder}")


        audio_files = sorted(input_folder.glob("*.mp4"))
        
        for audiofile in audio_files:
            wav_name = audiofile.stem + ".wav"
            wavfile = output_folder / wav_name
            convert_file(audiofile, wavfile)
            total += 1

    print(f"\nDone. Converted: {total} files.")

if __name__ == "__main__":
    main()