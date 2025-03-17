import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

class LJSpeechDataset(Dataset):
    def __init__(self, root_dir: str, segment_length: int = 8192):
        self.root_dir = Path(root_dir)
        self.segment_length = segment_length
        self.files = list(self.root_dir.glob("*.wav"))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        waveform, _ = torchaudio.load(audio_path)
        
        if waveform.size(1) >= self.segment_length: # Random crop if audio is longer than segment_length
            max_start = waveform.size(1) - self.segment_length
            start = torch.randint(0, max_start, (1,))
            waveform = waveform[:, start:start + self.segment_length]
        else: # Pad if audio is shorter
            pad_length = self.segment_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
        return waveform
