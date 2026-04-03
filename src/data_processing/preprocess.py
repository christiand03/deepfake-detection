import os
import glob
import torch
import torchaudio
import numpy as np
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, AutoProcessor

class MultimodalPreprocessor:
    def __init__(self, 
                 video_model_id="MCG-NJU/videomae-base", 
                 audio_model_id="MIT/ast-finetuned-audioset-10-10-0.4593", 
                 num_frames=16):
        
        print("Load Hugging Face Processors...")
        # Prozessor für die Video-Frames
        self.video_processor = AutoImageProcessor.from_pretrained(video_model_id)
        
        # Prozessor für Audio
        self.audio_processor = AutoProcessor.from_pretrained(audio_model_id)
        
        self.num_frames = num_frames
        
    def extract_video_tensor(self, video_path):
        """Extrahiert gleichmäßig verteilte Frames aus dem Video und wandelt sie in Tensoren um."""

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        # Logik damit Frames gleichmßig über Video verteilt werden
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy() # Shape: (num_frames, H, W, C)
        
        # Konvertiere Numpy-Liste für den HF Processor
        frames_list = list(frames)
        
        inputs = self.video_processor(frames_list, return_tensors="pt")
        
        # Entferne die Batch-Dimension
        return inputs["pixel_values"].squeeze(0)

    def extract_audio_tensor(self, video_path):
        """Liest die Audiospur direkt aus der Videodatei und wandelt sie in Features um."""

        waveform, sample_rate = torchaudio.load(video_path)
        
        # Stereo zu Mono konvertieren
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Sample Rate anpassen falls nötig
        target_sr = getattr(self.audio_processor.feature_extractor, 'sampling_rate', 16000)
        
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        
        inputs = self.audio_processor(waveform.squeeze().numpy(), sampling_rate=target_sr, return_tensors="pt")
        
        # Batch-Dimension entfernen
        return inputs["input_features"].squeeze(0) if "input_features" in inputs else inputs["input_values"].squeeze(0)

    def process_and_save(self, input_video_path, output_dir):
        """Verarbeitet eine Datei und speichert das Ergebnis als .pt Datei."""
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.splitext(os.path.basename(input_video_path))[0]
        output_path = os.path.join(output_dir, f"{filename}.pt")
        
        # Überspringen, falls schon verarbeitet
        if os.path.exists(output_path):
            print(f"Skip {filename}, if Data already exists.")
            return

        try:
            # Video verarbeiten
            video_tensor = self.extract_video_tensor(input_video_path)
            
            # Audio verarbeiten
            try:
                audio_tensor = self.extract_audio_tensor(input_video_path)
            except Exception as e:
                print(f"Warning: Audio for {filename} can not extracted. ({e})")
                audio_tensor = None
                
            # Daten in Dictionary 
            processed_data = {
                "video": video_tensor,
                "audio": audio_tensor
            }
            
            # Als PyTorch Tensor-Datei abspeichern
            torch.save(processed_data, output_path)
            print(f"Successfully processed and saved: {output_path}")
            
        except Exception as e:
            print(f"Error during processing of {input_video_path}: {e}")


if __name__ == "__main__":
    # Konfiguration der Pfade
    RAW_DATA_DIR = "Datasets/FaceForensics/original"
    PROCESSED_DATA_DIR = "data/processed_tensors"
    
    # Preprocessor initialisieren
    preprocessor = MultimodalPreprocessor(num_frames=16)
    
    # Alle .mp4 Dateien finden
    video_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.mp4"))
    print(f"Gefundene Videos: {len(video_files)}")
    
    # Schleife über alle Videos
    for video_path in video_files:
        preprocessor.process_and_save(video_path, PROCESSED_DATA_DIR)