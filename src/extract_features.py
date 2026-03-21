from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings("ignore")

# Paths
CLIPS_DIR = Path("data/clips")
OUT_CSV = Path("data/features.csv")

# Audio setting 
DEFAULT_SR = 22050 
N_MFCC = 13 # number of MFCC coefficients to extract from audio
RMS_THRESHOLD = 0.01
SUPPORTED_EXT = {".wav"}

def compute_features(audio_path: Path) -> dict | None:
    """
    Extract features from a single audio clip.
    Returns: dict of features for one clip or None if skipped

    """
    try:
        y, sr = librosa.load(audio_path, sr=DEFAULT_SR, mono=True)

    except Exception as e:
        print(f"WARNING: failed to load {audio_path}: {e}")
        return None
    
    #empty audio
    if y is None or len(y) == 0:
        print(f"WARNING: empty audio: {audio_path}")
        return None

    #frame-based features
    try:#[0] opening outer box to give inner numbers
        #how often waveform crosses zero amplitude
        zrc = librosa.feature.zero_crossing_rate(y=y)[0] 
        #perceived signal strength / loundness approx
        rms = librosa.feature.rms(y=y)[0] 
        #center of mass of spectrum (i.e. higher = brigher sound)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        #how spread out frequencies are around centroid
        spec_bandwith = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        #frequency below which large percentage of spectral energy lies i.e. brightness
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        #difference between peaks and valleys in spect i.e. harmonic peaks / flatter sound
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
        #energy mapped into 12 pitch class i.e. C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        #tonal centroid derived from harmonic relationships
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr) #harmonic = only harmonic component
        #summary of spectral envelope i.e. average each coefficient over time
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = N_MFCC)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr) #BPM, number of beats
    
    except Exception as e:
        print(f"WARNING: feature extraction failed for {audio_path}: {e}")
        return None

    # what about silence?
    mean_rms = float(np.mean(rms))
    if mean_rms < RMS_THRESHOLD:
        print(f"skip {audio_path} because silent")
        return None

    #all inforamtion into dictionary
    features = {}

    #basic data info
    features["filepath"] = str(audio_path)
    features["filename"] = audio_path.name
    features["label"] = audio_path.parent.name


    # scalar features
    features["tempo"] = float(tempo)
    features["zrc_mean"] = float(np.mean(zrc))
    features["zrc_std"] = float(np.std(zrc))
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["spectral_centroid_mean"] = float(np.mean(spec_centroid))
    features["spectral_centroid_std"] = float(np.std(spec_centroid))
    features["spectral_bandwith_mean"] = float(np.mean(spec_bandwith))
    features["spectral_bandwith_str"] = float(np.std(spec_bandwith))


    # spectral contrast        
    for i in range(spec_contrast.shape[0]):
        features[f"spectral_contrast_{i+1}_mean"] = float(np.mean(spec_contrast[i]))
        features[f"spectral_contrast_{i+1}_std"] = float(np.std(spec_contrast[i]))

    # Chroma (12)
    for i in range(chroma.shape[0]):
        features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

    # Tonnetz (6)
    for i in range(tonnetz.shape[0]):
        features[f"tonnetz_{i+1}_mean"] = float(np.mean(tonnetz[i]))
        features[f"tonnetz_{i+1}_std"] = float(np.std(tonnetz[i]))

    # MFCC (13)
    for i in range(mfcc.shape[0]):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    return features

def find_audiofiles(root:Path) -> list[Path]:
    """
    recursively find all audio files under root -> data/clips/'label'/'clip'.wav
    """
    files = []
    for path in root.rglob("*"): #recursively searching
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXT:
            files.append(path)
        
    return sorted(files)

def main():
    if not CLIPS_DIR.exists():
        raise FileNotFoundError(f"clips directory not found -> {CLIPS_DIR}")
    
    audio_files = find_audiofiles(CLIPS_DIR)
    if not audio_files:
        raise FileNotFoundError(f"audio files not found in {CLIPS_DIR}")    
    print(f"Found {len(audio_files)} clips")

    rows = []
    skipped = 0

    for i, audio_path in enumerate(audio_files, start = 1):
        results = compute_features(audio_path)
    
        if results is None:
            skipped += 1
            continue
        
        rows.append(results)
    
    if not rows:
        raise RuntimeError("no valid feature rows")


    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index = False)

    print(f"Saved {OUT_CSV} with {df.shape}")

if __name__ == "__main__":
    main()