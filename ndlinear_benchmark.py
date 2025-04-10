import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

from ndlinear import NdLinear

print("NdLinear successfully imported!")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Make sure results directory exists
os.makedirs("./results", exist_ok=True)

# Define a CNN with standard Linear layers
class StandardCNN(nn.Module):
    def __init__(self):
        super(StandardCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Initialize weights with kaiming initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

# Basic NdLinear CNN (original implementation)
class BasicNdLinearCNN(nn.Module):
    def __init__(self):
        super(BasicNdLinearCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Using NdLinear to reshape and maintain some structure
        self.fc1 = NdLinear(
            input_dims=(64, 12, 12),  # Reshape to preserve channel and spatial structure
            hidden_size=(32, 4, 1)    # Project to smaller dimensions
        )
        
        self.fc2 = NdLinear(
            input_dims=(32, 4, 1),
            hidden_size=(10, 1, 1)   # Output for 10 classes
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Reshape to preserve structure for NdLinear
        batch_size = x.size(0)
        x = x.permute(0, 1, 2, 3)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Reshape for output
        x = x.view(batch_size, 10)
        
        return F.log_softmax(x, dim=1)
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())

# Optimized NdLinear CNN architecture
class OptimizedNdLinearCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(OptimizedNdLinearCNN, self).__init__()
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Adjusted dropout rates
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 1.5)
        
        # Layer-specific structure preservation
        # Early layer: preserve spatial dimensions, reduce channels moderately
        self.fc1 = NdLinear(
            input_dims=(64, 12, 12),
            hidden_size=(48, 12, 12)
        )
        
        # Middle layer: maintain channels, reduce spatial dimensions
        self.fc2 = NdLinear(
            input_dims=(48, 12, 12),
            hidden_size=(48, 6, 6)
        )
        
        # Late layer: reduce both dimensions preparing for classification
        self.fc3 = NdLinear(
            input_dims=(48, 6, 6),
            hidden_size=(24, 3, 3)
        )
        
        # Final classification layer
        self.fc4 = NdLinear(
            input_dims=(24, 3, 3),
            hidden_size=(10, 1, 1)
        )
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Convolutional feature extraction with batch normalization
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Reshape to preserve structure for NdLinear
        batch_size = x.size(0)
        x = x.permute(0, 1, 2, 3)  # (batch, channels, height, width)
        
        # Layer-specific NdLinear transformations
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc4(x)
        x = x.reshape(batch_size, 10)
        
        return F.log_softmax(x, dim=1)
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# Structure-preserving CNN (original implementation)
class StructurePreservingCNN(nn.Module):
    def __init__(self):
        super(StructurePreservingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Structure-preserving NdLinear approach
        self.fc1 = NdLinear(
            input_dims=(12, 12, 64),  # Preserve spatial structure
            hidden_size=(8, 8, 32)    # Reduce dimensions while keeping structure
        )
        
        self.fc2 = NdLinear(
            input_dims=(8, 8, 32),
            hidden_size=(10, 1, 1)    # Output for 10 classes
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Reshape to preserve structure - from (batch, 64, 12, 12) to (batch, 12, 12, 64)
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)
        
        # Apply structure-preserving NdLinear
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Reshape for output
        x = x.reshape(batch_size, 10)
        
        return F.log_softmax(x, dim=1)
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())

def load_mnist(batch_size=64, val_split=0.1):
    """Load MNIST dataset with validation split"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full training dataset
    full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Split into training and validation
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Test dataset
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_lr_scheduler(optimizer, warmup_epochs=5, decay_factor=0.1, decay_epochs=[30, 60, 90]):
    """Create learning rate scheduler with warmup and decay"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Gradual warmup
            return (epoch + 1) / warmup_epochs
        
        # Step decay after warmup
        decay = decay_factor ** sum(epoch >= e for e in decay_epochs)
        return decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(model, train_loader, val_loader, optimizer, scheduler=None, epochs=10, patience=5):
    """Train the model with validation-based early stopping"""
    model.train()
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_times = []
    
    best_val_accuracy = 0
    patience_counter = 0
    best_model_weights = None
    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Step scheduler if provided
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'Current learning rate: {current_lr:.6f}')
        
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        
        train_losses.append(avg_loss)
        train_times.append(epoch_time)
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch}: Train loss: {avg_loss:.4f}, accuracy: {train_accuracy:.2f}%, '
              f'Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.2f}%, Time: {epoch_time:.2f}s')
        
        # Early stopping check
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                # Restore best model
                model.load_state_dict(best_model_weights)
                break
    
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    
    return train_losses, val_losses, val_accuracies, train_times

def validate(model, val_loader):
    """Validate the model"""
    model.eval()
    val_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    
    return val_loss, accuracy

def test(model, test_loader):
    """Test the model"""
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        start_time = time.time()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        test_time = time.time() - start_time
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    print(f'Inference time: {test_time:.2f}s')
    
    return test_loss, accuracy, test_time, all_preds, all_targets

def distillation_loss(student_logits, teacher_logits, true_labels, alpha=0.5, temperature=2.0):
    """
    Compute the knowledge distillation loss
    
    Args:
        student_logits: Raw logits from student model (before softmax)
        teacher_logits: Raw logits from teacher model (before softmax)
        true_labels: Ground truth labels
        alpha: Weight of soft targets loss vs hard targets loss (0-1)
        temperature: Temperature for softening probability distributions
        
    Returns:
        Combined distillation loss
    """
    # Hard targets loss - standard cross-entropy with true labels
    hard_loss = F.nll_loss(F.log_softmax(student_logits, dim=1), true_labels)
    
    # Soft targets - distilling teacher's knowledge
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    # Combined loss
    loss = alpha * soft_loss + (1 - alpha) * hard_loss
    return loss

def train_with_distillation(student_model, teacher_model, train_loader, val_loader, optimizer, 
                           scheduler=None, epochs=10, patience=5, temperature=2.0, alpha=0.5):
    """Train the student model with knowledge distillation from teacher"""
    student_model.train()
    teacher_model.eval()  # Teacher model in eval mode
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_times = []
    
    best_val_accuracy = 0
    patience_counter = 0
    best_model_weights = None
    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_output = teacher_model(data)
            
            # Train student with distillation
            optimizer.zero_grad()
            student_output = student_model(data)
            
            # Calculate distillation loss
            loss = distillation_loss(
                student_output,
                teacher_output,
                target,
                alpha=alpha,
                temperature=temperature
            )
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = student_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Step scheduler if provided
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'Current learning rate: {current_lr:.6f}')
        
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        
        train_losses.append(avg_loss)
        train_times.append(epoch_time)
        
        # Validate
        val_loss, val_accuracy = validate(student_model, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch}: Train loss: {avg_loss:.4f}, accuracy: {train_accuracy:.2f}%, '
              f'Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.2f}%, Time: {epoch_time:.2f}s')
        
        # Early stopping check
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_weights = student_model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                # Restore best model
                student_model.load_state_dict(best_model_weights)
                break
    
    if best_model_weights:
        student_model.load_state_dict(best_model_weights)
    
    return train_losses, val_losses, val_accuracies, train_times

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'./results/{model_name.replace(" ", "_").lower()}_confusion_matrix.png')
    plt.close()

def plot_learning_curves(train_losses, val_losses, val_accuracies, model_name):
    """Plot learning curves"""
    epochs = range(1, len(train_losses) + 1)
    
    # Plot loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./results/{model_name.replace(" ", "_").lower()}_loss_curves.png')
    plt.close()
    
    # Plot accuracy curve
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, val_accuracies, 'go-', label='Validation accuracy')
    plt.title(f'Validation Accuracy - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./results/{model_name.replace(" ", "_").lower()}_accuracy_curve.png')
    plt.close()

def plot_metrics_comparison(metrics, model_names, metric_name, title, ylabel, filename):
    """Plot comparison of metrics"""
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(model_names)), metrics, width=0.6)
    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if metric_name == 'params':
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:,}', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'./results/{filename}.png')
    plt.close()

def create_summary_table(model_names, accuracies, param_counts, train_times, test_times):
    """Create and save summary table"""
    # Calculate relative improvements compared to standard model
    param_reductions = [(param_counts[0] - count) / param_counts[0] * 100 for count in param_counts]
    accuracy_changes = [acc - accuracies[0] for acc in accuracies]
    train_speedups = [train_times[0] / time for time in train_times]
    inference_speedups = [test_times[0] / time for time in test_times]
    
    # Create summary table
    summary = pd.DataFrame({
        'Model': model_names,
        'Accuracy (%)': accuracies,
        'Parameters': param_counts,
        'Parameter Reduction (%)': [0] + param_reductions[1:],
        'Training Time (s)': train_times,
        'Training Speedup': [1.0] + train_speedups[1:],
        'Inference Time (s)': test_times,
        'Inference Speedup': [1.0] + inference_speedups[1:]
    })
    
    # Save summary to CSV
    summary.to_csv('./results/summary.csv', index=False)
    
    return summary

def create_enhanced_visualizations(results_dir='./results'):
    """Create enhanced visualizations for the benchmark results"""
    os.makedirs(results_dir, exist_ok=True)
    
    # Load summary data from CSV
    summary_df = pd.read_csv(f'{results_dir}/summary.csv')
    
    # 1. Create parameter reduction vs accuracy plot
    plt.figure(figsize=(12, 8))
    
    # Create size of bubbles based on parameter count (log scale for visibility)
    sizes = [np.log10(p) * 100 for p in summary_df['Parameters']]
    
    # Create scatter plot with custom colors
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(summary_df)]
    scatter = plt.scatter(summary_df['Parameters'], summary_df['Accuracy (%)'], 
               s=sizes, 
               alpha=0.7,
               c=colors)
    
    # Add model names as annotations
    for i, model in enumerate(summary_df['Model']):
        plt.annotate(model, 
                    (summary_df['Parameters'][i], summary_df['Accuracy (%)'][i]),
                    xytext=(10, 5),
                    textcoords='offset points',
                    fontsize=11,
                    fontweight='bold')
    
    # Set logarithmic x-scale to better visualize the parameter difference
    plt.xscale('log')
    
    # Add gridlines and styling
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Parameters (log scale)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs. Model Size Trade-off', fontsize=14, fontweight='bold')
    
    # Highlight the "sweet spot" area
    min_params = summary_df['Parameters'].min()
    max_params = summary_df['Parameters'].max()
    sweet_spot_x = np.array([min_params, min_params * 10])
    sweet_spot_y = np.array([97, 99])
    plt.fill_between(sweet_spot_x, sweet_spot_y[0], sweet_spot_y[1], 
                    color='green', alpha=0.1, label='Optimal Region')
    
    plt.legend(['Optimal Region', 'Models'])
    plt.tight_layout()
    plt.savefig(f'{results_dir}/accuracy_vs_parameters.png', dpi=300)
    plt.close()
    
    # 2. Create radar chart comparing models
    # Prepare normalized metrics for radar chart (higher is better)
    metrics = {
        'Accuracy': summary_df['Accuracy (%)'] / 100,
        'Param. Efficiency': 1 - (summary_df['Parameters'] / summary_df['Parameters'].max()),
        'Training Speed': 1 - (summary_df['Training Time (s)'] / summary_df['Training Time (s)'].max()),
        'Inference Speed': 1 - (summary_df['Inference Time (s)'] / summary_df['Inference Time (s)'].max())
    }
    
    categories = list(metrics.keys())
    N = len(categories)
    
    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add each model to the chart
    for i, model in enumerate(summary_df['Model']):
        values = [metrics[metric][i] for metric in categories]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=model)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    plt.title('Model Performance Comparison', fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/model_radar_chart.png', dpi=300)
    plt.close()
    
    print(f"Enhanced visualizations created in {results_dir} directory")

def main():
    # Set parameters
    batch_size = 64
    base_epochs = 2  # For standard model
    extended_epochs = 10  # For optimized models
    learning_rate = 0.001
    
    # Define models to test
    model_classes = [
        StandardCNN, 
        BasicNdLinearCNN, 
        StructurePreservingCNN,
        OptimizedNdLinearCNN
    ]
    
    model_names = [
        "Standard CNN", 
        "Basic NdLinear CNN", 
        "Structure-Preserving CNN",
        "Optimized NdLinear CNN"
    ]
    
    # Load MNIST dataset with validation split
    print("\nLoading MNIST dataset with validation split...")
    train_loader, val_loader, test_loader = load_mnist(batch_size)
    
    # Storage for results
    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []
    all_accuracies = []
    all_param_counts = []
    all_train_times = []
    all_test_times = []
    all_predictions = []
    all_targets = []
    
    standard_model = None  # Will store the standard model for distillation
    
    # Train and evaluate each model
    for i, (model_class, name) in enumerate(zip(model_classes, model_names)):
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print(f"{'='*50}")
        
        # Create model
        if model_class == OptimizedNdLinearCNN:
            model = model_class(dropout_rate=0.3).to(device)
        else:
            model = model_class().to(device)
        
        # Get parameter count
        param_count = model.get_param_count()
        print(f"Model parameter count: {param_count:,}")
        
        # Determine epochs based on model
        if name == "Standard CNN":
            epochs = base_epochs
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = None
            
            # Train using standard approach
            train_losses, val_losses, val_accuracies, total_train_time = train(
                model, train_loader, val_loader, optimizer, scheduler, epochs=epochs
            )
            
            # Store standard model for distillation
            standard_model = model
            
        elif name == "Optimized NdLinear CNN" and standard_model is not None:
            # Use knowledge distillation for optimized model
            epochs = extended_epochs
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate/2, weight_decay=1e-4)
            scheduler = get_lr_scheduler(optimizer, warmup_epochs=2, decay_epochs=[5, 8])
            
            print("Training with knowledge distillation from Standard CNN...")
            train_losses, val_losses, val_accuracies, total_train_time = train_with_distillation(
                model, standard_model, train_loader, val_loader, optimizer, scheduler, 
                epochs=epochs, temperature=2.0, alpha=0.5
            )
            
        else:
            # Use enhanced training for other NdLinear models
            epochs = extended_epochs
            optimizer = optim.Adam(model.parameters(), lr=learning_rate/2)
            scheduler = get_lr_scheduler(optimizer, warmup_epochs=2, decay_epochs=[5, 8])
            
            train_losses, val_losses, val_accuracies, total_train_time = train(
                model, train_loader, val_loader, optimizer, scheduler, epochs=epochs
            )
        
        # Test model
        test_loss, accuracy, test_time, predictions, targets = test(model, test_loader)
        
        # Plot learning curves
        plot_learning_curves(train_losses, val_losses, val_accuracies, name)
        
        # Plot confusion matrix
        plot_confusion_matrix(targets, predictions, name)
        
        # Store results
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_val_accuracies.append(val_accuracies)
        all_accuracies.append(accuracy)
        all_param_counts.append(param_count)
        all_train_times.append(sum(total_train_time))
        all_test_times.append(test_time)
        all_predictions.append(predictions)
        all_targets.append(targets)
        
        print(f"Training and evaluation completed for {name}")
    
    # Plot accuracy comparison
    plot_metrics_comparison(all_accuracies, model_names, 'accuracy', 
                          'Accuracy Comparison', 'Accuracy (%)', 'accuracy_comparison')
    
    # Plot parameter count comparison
    plot_metrics_comparison(all_param_counts, model_names, 'params',
                          'Parameter Count Comparison', 'Number of Parameters', 'parameter_comparison')
    
    # Plot training time comparison
    plot_metrics_comparison(all_train_times, model_names, 'train_time',
                          'Training Time Comparison', 'Training Time (s)', 'training_time_comparison')
    
    # Plot inference time comparison
    plot_metrics_comparison(all_test_times, model_names, 'test_time',
                          'Inference Time Comparison', 'Inference Time (s)', 'inference_time_comparison')
    
    # Create summary table
    summary = create_summary_table(model_names, all_accuracies, all_param_counts, 
                                all_train_times, all_test_times)
    
    # Create enhanced visualizations
    create_enhanced_visualizations()
    
    print("\nBenchmark completed! Results saved to './results/' directory")
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()