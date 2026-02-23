import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.config import HyperMLPConfig
import wandb
import time

class BaselineModel(nn.Module):
    """
    Standard autoregressive Transformer model to serve as a baseline.
    Uses PyTorch's built-in TransformerEncoder with a causal mask.
    """
    def __init__(self, config: HyperMLPConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # Standard Transformer block: Self-Attention + FFN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        # Create causal mask ensuring future tokens are masked out
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
        
        x = self.embeddings(input_ids)
        x = self.encoder(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def main():
    # 1. Configuration (Must exactly match HyperMLP main.py)
    config = HyperMLPConfig(
        d_model=128,
        n_heads=2,
        max_seq_len=64,
        d_qk=32,
        d_vo=64,
        rank_s=8,
        use_hyperglu=True,
        batch_size=8,
        vocab_size=1000,
        n_layers=2
    )

    print("Baseline Configuration:", config)

    # Initialize wandb
    wandb.init(project="hypermlp", name="baseline-transformer-run", config=config.__dict__)

    # 2. Model Initialization
    model = BaselineModel(config)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Baseline Model initialized successfully with {param_count:,} parameters.")
    wandb.config.update({"param_count": param_count})

    # 3. Toy Dataset (same counting pattern)
    vocab_size = config.vocab_size
    batch_size = config.batch_size
    seq_len = config.max_seq_len
    num_samples = 150  # 100 train, 25 val, 25 test

    offsets = torch.randint(0, vocab_size, (num_samples, 1))
    indices = torch.arange(seq_len + 1).unsqueeze(0).repeat(num_samples, 1)
    raw_data = (indices + offsets) % vocab_size
    
    x_train, y_train = raw_data[:100, :-1], raw_data[:100, 1:]
    x_val, y_val = raw_data[100:125, :-1], raw_data[100:125, 1:]
    x_test, y_test = raw_data[125:, :-1], raw_data[125:, 1:]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # 4. Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    model.train()
    print("Starting training loop...")
    
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_time = time.time()
            optimizer.zero_grad()
            
            outputs = model(data)
            
            # Reshape for loss: (B*T, V) vs (B*T)
            loss = criterion(outputs.view(-1, vocab_size), target.view(-1))
            
            predictions = torch.argmax(outputs, dim=-1)
            accuracy = (predictions.view(-1) == target.view(-1)).float().mean().item()
            perplexity = torch.exp(loss).item()
            
            loss.backward()
            optimizer.step()
            
            batch_time = time.time() - batch_start_time
            steps_per_sec = 1.0 / batch_time if batch_time > 0 else 0
            
            total_loss += loss.item()
            total_acc += accuracy
            
            wandb.log({
                "loss": loss.item(),
                "accuracy": accuracy,
                "perplexity": perplexity,
                "steps_per_sec": steps_per_sec,
                "epoch": epoch,
                "batch": batch_idx
            })

            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}, PPL: {perplexity:.4f}, Steps/sec: {steps_per_sec:.2f}")

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Average Training Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        wandb.log({"avg_epoch_loss": avg_loss, "avg_epoch_acc": avg_acc, "epoch": epoch})

        # Validation Step
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for val_data, val_target in val_loader:
                    outputs = model(val_data)
                    loss = criterion(outputs.view(-1, vocab_size), val_target.view(-1))
                    val_loss += loss.item()
                    predictions = torch.argmax(outputs, dim=-1)
                    val_acc += (predictions.view(-1) == val_target.view(-1)).float().mean().item()
            if len(val_loader) > 0:
                avg_val_loss = val_loss / len(val_loader)
                avg_val_acc = val_acc / len(val_loader)
                val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
                print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}, PPL: {val_perplexity:.4f}")
                wandb.log({"val_loss": avg_val_loss, "val_accuracy": avg_val_acc, "val_perplexity": val_perplexity, "epoch": epoch})
            model.train()

    print("Evaluating on test set...")
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for test_data, test_target in test_loader:
            outputs = model(test_data)
            loss = criterion(outputs.view(-1, vocab_size), test_target.view(-1))
            test_loss += loss.item()
            predictions = torch.argmax(outputs, dim=-1)
            test_acc += (predictions.view(-1) == test_target.view(-1)).float().mean().item()
    if len(test_loader) > 0:
        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_acc / len(test_loader)
        test_perplexity = torch.exp(torch.tensor(avg_test_loss)).item()
        print(f"Test Loss: {avg_test_loss:.4f}, Acc: {avg_test_acc:.4f}, PPL: {test_perplexity:.4f}")
        wandb.log({"test_loss": avg_test_loss, "test_accuracy": avg_test_acc, "test_perplexity": test_perplexity})

    print("Training finished. Saving model...")
    torch.save(model.state_dict(), "models/baseline/baseline_model.pt")
    wandb.save("models/baseline/baseline_model.pt")
    
    wandb.finish()

if __name__ == "__main__":
    main()
