import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from data.audio import LJSpeechDataset
from models.codec import VQVAE

def train_vqvae():
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VQVAE(
        in_channels=1,  # mono audio
        embedding_dim=64,
        num_embeddings=512,  # codebook size
        hidden_dims=[32, 64, 128, 256]
    ).to(device)
    
    dataset = LJSpeechDataset("ljspeech_raw", segment_length=8192)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_vq = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            x = batch.to(device)
            
            x_recon, vq_loss, vq_info = model(x)
            
            recon_loss = torch.mean((x - x_recon) ** 2)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'vq': vq_loss.item(),
                'perplexity': vq_info['perplexity'].item()
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_vq = total_vq / len(dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Reconstruction Loss: {avg_recon:.4f}")
        print(f"Average VQ Loss: {avg_vq:.4f}")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path("checkpoints")
            checkpoint_path.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path / f"vqvae_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train_vqvae()
