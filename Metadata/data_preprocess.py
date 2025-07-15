import pandas as pd
import json
import torch
import os
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ========================================
# STEP 1: LOAD DATA
# ========================================
def load_data(json_file_path):
    """Load data from JSON file and create DataFrame"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    
    for url, info in data.items():
        texts.append(info['content'])
        labels.append(info['category'])
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    
    # Label encoding
    label_to_id = {
        'sport': 0,
        'bus_fin': 1, 
        'technology': 2,
        'health_medical': 3
    }
    
    df['label_id'] = df['label'].map(label_to_id)
    
    return df, label_to_id

# ========================================
# STEP 2: STRATIFIED SPLIT
# ========================================
def split_data(df, test_size=0.3, val_size=0.5, random_state=42, save_to_json=True):
    """Split data into train/val/test with stratification"""
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['label_id'], 
        random_state=random_state
    )
    
    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=val_size, 
        stratify=temp_df['label_id'], 
        random_state=random_state
    )
    
    # Print split information
    print("=== DATA SPLIT SUMMARY ===")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify stratification worked
    print("\n=== CLASS DISTRIBUTION ===")
    print("Original distribution:")
    print(df['label'].value_counts(normalize=True).round(3))
    
    print("\nTrain distribution:")
    print(train_df['label'].value_counts(normalize=True).round(3))
    
    print("\nValidation distribution:")
    print(val_df['label'].value_counts(normalize=True).round(3))
    
    print("\nTest distribution:")
    print(test_df['label'].value_counts(normalize=True).round(3))
    
    # Save split data to JSON files
    if save_to_json:
        print("\n=== SAVING SPLIT DATA TO JSON FILES ===")
        
        # Convert DataFrames to JSON format
        train_json = {}
        val_json = {}
        test_json = {}
        
        for idx, row in train_df.iterrows():
            train_json[f"train_{idx}"] = {
                "text": row['text'],
                "label": row['label'],
                "label_id": row['label_id']
            }
        
        for idx, row in val_df.iterrows():
            val_json[f"val_{idx}"] = {
                "text": row['text'],
                "label": row['label'],
                "label_id": row['label_id']
            }
        
        for idx, row in test_df.iterrows():
            test_json[f"test_{idx}"] = {
                "text": row['text'],
                "label": row['label'],
                "label_id": row['label_id']
            }
        
        # Save to files
        with open('train_data.json', 'w', encoding='utf-8') as f:
            json.dump(train_json, f, ensure_ascii=False, indent=2)
        
        with open('val_data.json', 'w', encoding='utf-8') as f:
            json.dump(val_json, f, ensure_ascii=False, indent=2)
        
        with open('test_data.json', 'w', encoding='utf-8') as f:
            json.dump(test_json, f, ensure_ascii=False, indent=2)
        
        print("✅ Saved split data to:")
        print("   - train_data.json")
        print("   - val_data.json") 
        print("   - test_data.json")
    
    return train_df, val_df, test_df

# ========================================
# STEP 3: ROBERTA TOKENIZATION
# ========================================
class NewsDataset(Dataset):
    """Custom Dataset class for RoBERTa tokenization"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # RoBERTa tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_tokenized_datasets(train_df, val_df, test_df, model_name='roberta-large', max_length=512):
    """Create tokenized datasets using RoBERTa tokenizer"""
    
    # Initialize RoBERTa tokenizer
    print(f"Loading {model_name} tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = NewsDataset(
        train_df['text'].tolist(), 
        train_df['label_id'].tolist(), 
        tokenizer, 
        max_length
    )
    
    val_dataset = NewsDataset(
        val_df['text'].tolist(), 
        val_df['label_id'].tolist(), 
        tokenizer, 
        max_length
    )
    
    test_dataset = NewsDataset(
        test_df['text'].tolist(), 
        test_df['label_id'].tolist(), 
        tokenizer, 
        max_length
    )
    
    print(f"✅ Tokenization complete!")
    print(f"   - Max sequence length: {max_length}")
    print(f"   - Train samples: {len(train_dataset)}")
    print(f"   - Val samples: {len(val_dataset)}")
    print(f"   - Test samples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, tokenizer

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    """Create DataLoaders for training"""
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"✅ DataLoaders created with batch_size={batch_size}")
    
    return train_loader, val_loader, test_loader

# ========================================
# MAIN EXECUTION
# ========================================
def main():
    # Configuration - Update this path to match your file location
    JSON_FILE_PATH = 'Metadata\metadata.json'  # Change this to your actual file path
    BATCH_SIZE = 16
    MAX_LENGTH = 512
    
    try:
        # Step 1: Load data
        print("=== STEP 1: LOADING DATA ===")
        print(f"Looking for file: {JSON_FILE_PATH}")
        df, label_to_id = load_data(JSON_FILE_PATH)
        print(f"✅ Loaded {len(df)} samples")
        
        # Step 2: Split data
        print("\n=== STEP 2: SPLITTING DATA ===")
        train_df, val_df, test_df = split_data(df)
        
        # Step 3: Tokenize with RoBERTa
        print("\n=== STEP 3: TOKENIZING WITH ROBERTA ===")
        train_dataset, val_dataset, test_dataset, tokenizer = create_tokenized_datasets(
            train_df, val_df, test_df, max_length=MAX_LENGTH
        )
        
        # Step 4: Create DataLoaders
        print("\n=== STEP 4: CREATING DATALOADERS ===")
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE
        )
        
        # Test tokenization (optional)
        print("\n=== TOKENIZATION TEST ===")
        sample_batch = next(iter(train_loader))
        print(f"Input IDs shape: {sample_batch['input_ids'].shape}")
        print(f"Attention mask shape: {sample_batch['attention_mask'].shape}")
        print(f"Labels shape: {sample_batch['labels'].shape}")
        
        print("\n🎉 SUCCESS! Data is ready for RoBERTa fine-tuning!")
        
        return train_loader, val_loader, test_loader, tokenizer, label_to_id
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("💡 Solutions:")
        print("   1. Make sure 'metadata.json' exists in the same folder as your script")
        print("   2. Or update JSON_FILE_PATH to the correct path")
        print("   3. Current working directory:", os.getcwd())
        return None, None, None, None, None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None, None, None

# Run the main function
if __name__ == "__main__":
    result = main()
    if result[0] is not None:
        train_loader, val_loader, test_loader, tokenizer, label_to_id = result
        print("✅ All components ready for training!")
    else:
        print("❌ Setup failed. Please check the errors above.")