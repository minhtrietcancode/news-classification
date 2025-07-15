import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm # For progress bars in Colab

# --- 1. Data Loading ---

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(filepath):
    """Loads data from a JSON file and extracts texts and labels."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    texts = []
    labels = []
    for key in data:
        texts.append(data[key]['text'])
        labels.append(data[key]['label_id'])
    return texts, labels

# Load datasets
print("Loading datasets...")
train_texts, train_labels = load_data('train_val_test/train_data.json')
val_texts, val_labels = load_data('train_val_test/val_data.json')
test_texts, test_labels = load_data('train_val_test/test_data.json')

# Display dataset statistics
print(f"Train samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_texts)}")

# Determine the number of unique labels (classes)
unique_labels = sorted(list(set(train_labels + val_labels + test_labels)))
num_classes = len(unique_labels)
print(f"Number of classes: {num_classes}")

# Create a mapping from label_id to label name for clarity in confusion matrix
# Assuming label_id 0 -> 'sport', 1 -> 'bus_fin', 2 -> 'technology', 3 -> 'health_medical' based on attached files
# You might need to adjust this if your label_ids map differently.
label_id_to_name = {
    0: 'sport',
    1: 'bus_fin',
    2: 'technology',
    3: 'health_medical'
}
class_names = [label_id_to_name[i] for i in range(num_classes)]


# --- 2. Model Setup ---

PRE_TRAINED_MODEL_NAME = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=num_classes)

# Set up for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")


# --- 3. Training Configuration ---

MAX_LEN = 256  # Max sequence length for RoBERTa
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

train_dataset = NewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = NewsDataset(val_texts, val_labels, tokenizer, MAX_LEN)
test_dataset = NewsDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss().to(device)


# --- 4. Training Process ---

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler=None):
    model = model.train()
    losses = []
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.numel()

        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / total_samples, sum(losses) / len(losses)

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.numel()
            losses.append(loss.item())

    return correct_predictions.double() / total_samples, sum(losses) / len(losses)


print("\nStarting training...")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device
    )
    print(f"Train loss {train_loss:.4f} accuracy {train_acc:.4f}")

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device
    )
    print(f"Val   loss {val_loss:.4f} accuracy {val_acc:.4f}")

# Save the trained model
output_dir = './model_save/'
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\nModel saved to {output_dir}")


# --- 5. Evaluation ---

print("\nEvaluating model on test data...")
y_pred_list = []
y_true_list = []
model = model.eval()

with torch.no_grad():
    for batch in tqdm(test_data_loader, desc="Testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        y_pred_list.extend(preds.cpu().numpy())
        y_true_list.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(y_true_list, y_pred_list)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Generate and plot confusion matrix
cm = confusion_matrix(y_true_list, y_pred_list)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig("model_matrix.png")
