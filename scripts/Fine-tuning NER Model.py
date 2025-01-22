import os
from datasets import load_dataset, DatasetDict
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split

# Step 1: Model Selection
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)

# Step 2: Dataset Loading
def load_conll_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        tokens, labels = [], []
        for line in lines:
            if line.strip():
                parts = line.split('\t')
                if len(parts) == 2:
                    tokens.append(parts[0])
                    labels.append(parts[1].strip())
            else:
                if tokens:  # End of a sentence
                    dataset.append((tokens, labels))
                    tokens, labels = [], []
    return dataset

# Load the dataset
data = load_conll_dataset(r"C:\Users\User\Desktop\New folder (3)\labeled_data_all_channels.conll")
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

# Convert to DatasetDict
train_dataset = DatasetDict({'train': [{'tokens': tokens, 'labels': labels} for tokens, labels in train_data]})
val_dataset = DatasetDict({'validation': [{'tokens': tokens, 'labels': labels} for tokens, labels in val_data]})

# Step 3: Tokenization and Label Alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, padding='max_length', max_length=128)
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(tokenized_inputs['input_ids'][i])  # -100 is used to ignore certain tokens
        for j, label_id in enumerate(label):
            if word_ids[j] is not None:  # Only label the first token of the word
                label_ids[word_ids[j]] = label_id
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Tokenize the datasets
train_tokenized = train_dataset['train'].map(tokenize_and_align_labels, batched=True)
val_tokenized = val_dataset['validation'].map(tokenize_and_align_labels, batched=True)

# Step 4: Training Configuration
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Step 5: Model Training
model = XLMRobertaForTokenClassification.from_pretrained(model_name, num_labels=len(set(label for _, labels in train_data for label in labels)))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
)

trainer.train()

# Step 6: Performance Evaluation
trainer.evaluate()

# Step 7: Model Saving
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
