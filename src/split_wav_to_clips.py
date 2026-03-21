from __future__ import annotations
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

WAV_DIR = Path("data/wav")
CLIPS_DIR = Path("data/clips")

LABELS = ["bach", "beethoven", "schumann"]

CLIP_TIME = 10 # seconds
HOP_TIME = 10 #seconds
DEFAULT_SR = 22050

RMS_THRESHOLD = 0.01
PEAK_THRESHOLD = 0.03

def is_clip_quiet(y:np.ndarray) -> bool:
    if len(y) == 0:
        return True
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y)) #maximum of absolute value of every element

    return rms < RMS_THRESHOLD or peak < PEAK_THRESHOLD

def split_file(input_path : Path, output_folder: Path) -> int:
    y, sr = librosa.load(input_path, sr=DEFAULT_SR, mono=True)

    clip_length = CLIP_TIME * sr
    hop_length = HOP_TIME * sr

    saved : int = 0
    stem = input_path.stem

    for start in range(0, len(y)-int(clip_length)+1, int(hop_length)):
        end = start + clip_length
        clip = y[start:end]

        if is_clip_quiet(clip):
            continue
        out_name = f"{stem}_clip_{saved:03d}.wav"
        out_path = output_folder / out_name

        sf.write(out_path, clip, sr)
        saved += 1
    
    return saved

def main() -> None:
    total_saved = 0
    for label in LABELS:
        input_folder = WAV_DIR/label
        output_folder = CLIPS_DIR/label

        output_folder.mkdir(parents = True, exist_ok = True)

        wav_files = sorted(input_folder.glob("*.wav"))

        if not wav_files:
            print(f"no WAV file found")
            continue
        for wavfile in wav_files:
            count = split_file(wavfile, output_folder)
            print(f"{wavfile.name}: saved {count} clips")
            total_saved += count
    print(f"\nDone. Total saved clips: {total_saved}")

if __name__ == "__main__":
    main()
    
