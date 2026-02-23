import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.config import HyperMLPConfig
from src.model import HyperMLPModel
import wandb
import time

def main():
    # 1. Configuration
    config = HyperMLPConfig(
        d_model=128,      # Smaller dimension for toy example
        n_heads=2,
        max_seq_len=64,   # Short sequence length for testing
        d_qk=32,          # 128 / (2 * 2) = 32
        d_vo=64,          # 128 / 2 = 64
        rank_s=8,
        use_hyperglu=True,
        batch_size=8,
        vocab_size=1000,
        n_layers=2
    )
    
    print("Configuration:", config)

    # Initialize wandb
    wandb.init(project="hypermlp", name="hypermlp-run", config=config.__dict__)

    # 2. Model Initialization
    try:
        model = HyperMLPModel(config)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized successfully with {param_count:,} parameters.")
        wandb.config.update({"param_count": param_count})
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Please implement the skeletons in src/ first.")
        return

    # 3. Toy Dataset (Language Modeling task: predict next token)
    vocab_size = 1000
    # Add head to model if not present in skeleton (standard LM head)
    # Ideally this should be inside HyperMLPModel, but adding here if needed for demo
    if not hasattr(model, 'head'):
        model.head = nn.Linear(config.d_model, vocab_size, bias=False)
        print("Added temporary LM head to model.")

    batch_size = 8
    seq_len = config.max_seq_len
    num_samples = 150  # 100 train, 25 val, 25 test

    # Toy data with a simple learnable pattern (counting sequence)
    # Each sequence is just [offset, offset+1, offset+2, ...] % vocab_size
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

    # 5. Training Loop Skeleton
    model.train()
    print("Starting training loop...")
    
    epochs = 50  # increased epochs to actually see training curve nicely
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_time = time.time()
            optimizer.zero_grad()
            
            # Forward pass
            # Note: The user needs to ensure their model.forward returns what's expected.
            # Usually: logits or (loss, logits)
            try:
                # Assuming model takes input_ids
                # We might need to handle embedding layer if not in model yet.
                # For this skeleton, we assume model.forward expects input_ids.
                outputs = model(data)
                
                # Retrieve logits
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Reshape for loss: (B*T, V) vs (B*T)
                loss = criterion(logits.view(-1, vocab_size), target.view(-1))
                
                predictions = torch.argmax(logits, dim=-1)
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
                    
            except NotImplementedError:
                print("Model forward pass not implemented yet. Skipping batch.")
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error during training step: {e}")
                break
        
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
                    try:
                        outputs = model(val_data)
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs
                        loss = criterion(logits.view(-1, vocab_size), val_target.view(-1))
                        val_loss += loss.item()
                        predictions = torch.argmax(logits, dim=-1)
                        val_acc += (predictions.view(-1) == val_target.view(-1)).float().mean().item()
                    except Exception as e:
                        print(f"Validation error: {e}")
                        break
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
            try:
                outputs = model(test_data)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                loss = criterion(logits.view(-1, vocab_size), test_target.view(-1))
                test_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                test_acc += (predictions.view(-1) == test_target.view(-1)).float().mean().item()
            except Exception as e:
                print(f"Test error: {e}")
                break
    if len(test_loader) > 0:
        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_acc / len(test_loader)
        test_perplexity = torch.exp(torch.tensor(avg_test_loss)).item()
        print(f"Test Loss: {avg_test_loss:.4f}, Acc: {avg_test_acc:.4f}, PPL: {test_perplexity:.4f}")
        wandb.log({"test_loss": avg_test_loss, "test_accuracy": avg_test_acc, "test_perplexity": test_perplexity})

    print("Training finished. Saving model...")
    torch.save(model.state_dict(), "models/hypermlp/hypermlp_model.pt")
    wandb.save("models/hypermlp/hypermlp_model.pt")
    
    wandb.finish()

if __name__ == "__main__":
    main()
