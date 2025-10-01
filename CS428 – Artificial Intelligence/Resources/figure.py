import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
def read_metrics_from_file(filename):
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    
    with open(filename, 'r') as file:
        for line in file:
            if 'loss:' in line:
                parts = line.split()
                train_loss.append(float(parts[parts.index('loss:') + 1]))
                train_accuracy.append(float(parts[parts.index('accuracy:') + 1]))
                val_loss.append(float(parts[parts.index('val_loss:') + 1]))
                val_accuracy.append(float(parts[parts.index('val_accuracy:') + 1]))
                
    return train_loss, train_accuracy, val_loss, val_accuracy

def plot_metrics(train_loss, train_accuracy, val_loss, val_accuracy):
    # Professional color palette
    color_palette = {"train": "#3498db", "val": "#e74c3c"}
    
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(11,7), facecolor="#ffffff")
    
    # Line plots
    plt.plot(epochs, train_accuracy, label='Training Accuracy', 
             color=color_palette["train"], linewidth=3.5)
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', 
             color=color_palette["val"], linewidth=3.5, linestyle='--')
    
    # Axes and label styling
    plt.xlabel('Epochs', fontsize=24, labelpad=15, color="#ffffff", fontweight="bold")
    plt.ylabel('Accuracy', fontsize=24, labelpad=15, color="#ffffff", fontweight="bold")
    # plt.title('Training and Validation Accuracy', fontsize=22, pad=20, color="#333333")
    plt.xticks(fontsize=24, color="#333333", fontweight="bold")
    plt.yticks(fontsize=24, color="#333333", fontweight="bold")
    
    # Axes range
    plt.xlim(0,75)
    plt.ylim(0,1)
    
    # Gridlines
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6, color="#cccccc")
    
    # Remove top and right spines for cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#aaaaaa')
    plt.gca().spines['left'].set_color('#aaaaaa')
    
    # Legend styling
    font = font_manager.FontProperties(family= 'Arial',  # 'Times new roman', 
                                   weight='bold',
                                   style='normal', size=24)
    # plt.legend(loc="lower right", fontsize=24, frameon=True, framealpha=0.9, edgecolor="#e0e0e0")
    plt.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="#e0e0e0", prop=font)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor="#ffffff")
    plt.show()

filename = 'eval_Res_f2.txt'
output_filename = filename + '.png'
train_loss, train_accuracy, val_loss, val_accuracy = read_metrics_from_file(filename)
plot_metrics(train_loss, train_accuracy, val_loss, val_accuracy)