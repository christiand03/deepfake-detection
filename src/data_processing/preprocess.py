import cv2
import os
import glob
import numpy as np
from moviepy import VideoFileClip
from tqdm import tqdm
import concurrent.futures
import multiprocessing

# Konfiguration
DATASET_ROOT = "Datasets/FaceForensics"
OUTPUT_BASE = "Processed_Dataset"
NUM_FRAMES = 16
TARGET_SIZE = (224, 224)

# Kategorien
REAL_CATEGORIES = ["original"]
FAKE_CATEGORIES = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

# CPU-Kerne für Multiprocessing (1 Kern wird frei gelassen, damit das System nicht komplett blockiert)
NUM_CORES = max(1, multiprocessing.cpu_count() - 1) 

def process_single_video_task(task_args):
    """
    Diese Funktion verarbeitet genau EIN Video. 
    Sie ist so geschrieben, dass sie auf einem isolierten CPU-Kern laufen kann.
    """
    video_path, category, label_folder = task_args
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    unique_video_name = f"{video_name}_{category}"
    
    # Zielpfade
    out_frames_dir = os.path.join(OUTPUT_BASE, "frames", label_folder)
    out_audio_dir = os.path.join(OUTPUT_BASE, "audio", label_folder)
    
    video_frame_dir = os.path.join(out_frames_dir, unique_video_name)
    audio_path = os.path.join(out_audio_dir, f"{unique_video_name}.wav")
    
    # SKIP LOGIK
    # Wenn der Ordner schon existiert und exakt 16 Bilder hat, sind wir hier fertig!
    if os.path.exists(video_frame_dir) and len(glob.glob(os.path.join(video_frame_dir, "*.jpg"))) == NUM_FRAMES:
        return True # Übersprungen

    os.makedirs(video_frame_dir, exist_ok=True)

    # 1. Audio Extrahieren
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is not None:
            clip.audio.write_audiofile(audio_path, logger=None, verbose=False)
        clip.close()
    except Exception:
        pass

    # Frames Extrahieren
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < NUM_FRAMES:
        cap.release()
        return False

    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    
    extracted_count = 0
    current_frame = 0
    
    while cap.isOpened() and extracted_count < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame == frame_indices[extracted_count]:
            frame_resized = cv2.resize(frame, TARGET_SIZE)
            frame_filename = os.path.join(video_frame_dir, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame_resized)
            extracted_count += 1
            
        current_frame += 1

    cap.release()
    return True


if __name__ == "__main__":
    print(f"Start Preprocessing with {NUM_CORES} CPU-Cores...")
    
    # Erstelle die Basis-Ordner
    for label in ["Real", "Fake"]:
        os.makedirs(os.path.join(OUTPUT_BASE, "frames", label), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE, "audio", label), exist_ok=True)

    all_categories = REAL_CATEGORIES + FAKE_CATEGORIES
    
    # Sammle alle Aufgaben in einer großen Liste
    tasks = []
    for category in all_categories:
        category_path = os.path.join(DATASET_ROOT, category)
        if not os.path.exists(category_path):
            continue
            
        search_pattern = os.path.join(category_path, "**", "*.mp4")
        video_files = glob.glob(search_pattern, recursive=True)
        
        label_folder = "Real" if category in REAL_CATEGORIES else "Fake"
        
        for video_path in video_files:
            tasks.append((video_path, category, label_folder))
            
    print(f"Found {len(tasks)} Videos. Starting processing...")

    # Verteile die Aufgaben auf die CPU-Kerne
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        # executor.map verteilt die Liste 'tasks' auf die Kerne. 
        list(tqdm(executor.map(process_single_video_task, tasks), total=len(tasks), desc="Processing Videos"))

    print("\nAll Data are prepared! Phase 0 is finished.")