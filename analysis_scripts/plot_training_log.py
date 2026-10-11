import matplotlib.pyplot as plt
import re
import sys
import os

def parse_log(log_file):
    epochs = []
    losses = []
    train_accs = []
    test_accs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Format: epoch:  0 ; loss:  1.23 ; train_acc:  0.5 ; test_acc:  0.4 ; epoch time:  0.1s
            # Or: epoch: 0 | loss: 1.23 | train_acc: 0.5 | test_acc: 0.4 | time: 0.1s
            if "loss:" in line and "train_acc:" in line:
                try:
                    # Normalize separators
                    line = line.replace(';', '|')
                    parts = line.split('|')
                    
                    epoch_str = [p for p in parts if "epoch" in p][0]
                    loss_str = [p for p in parts if "loss" in p][0]
                    train_acc_str = [p for p in parts if "train_acc" in p][0]
                    test_acc_str = [p for p in parts if "test_acc" in p][0]
                    
                    epoch = int(re.search(r'\d+', epoch_str).group())
                    loss = float(loss_str.split(':')[1].strip())
                    train_acc = float(train_acc_str.split(':')[1].strip())
                    test_acc = float(test_acc_str.split(':')[1].strip())
                    
                    epochs.append(epoch)
                    losses.append(loss)
                    train_accs.append(train_acc)
                    test_accs.append(test_acc)
                except Exception as e:
                    # print(f"Skipping line: {line.strip()} due to {e}")
                    continue
                    
    return epochs, losses, train_accs, test_accs

def plot_results(log_file, output_file="training_plot.png"):
    epochs, losses, train_accs, test_accs = parse_log(log_file)
    
    if not epochs:
        print(f"No valid training data found in {log_file}.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    ax1.plot(epochs, losses, label='Training Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Accuracy
    ax2.plot(epochs, train_accs, label='Train Acc', color='blue')
    ax2.plot(epochs, test_accs, label='Test Acc', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training_log.py <log_file> [output_png]")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "training_plot.png"
    plot_results(log_file, output_file)
