# Step 1: Install Required Libraries

# Step 2: Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (XLMRobertaTokenizerFast, XLMRobertaForTokenClassification,
                          DistilBertTokenizerFast, DistilBertForTokenClassification,
                          BertTokenizerFast, BertForTokenClassification,
                          Trainer, TrainingArguments)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Step 3: Dataset Loading Function
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

# Step 4: Load the Dataset
data = load_conll_dataset('labeled_data_all_channels.conll')
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_list([{'tokens': tokens, 'labels': labels} for tokens, labels in train_data])
val_dataset = Dataset.from_list([{'tokens': tokens, 'labels': labels} for tokens, labels in val_data])

# Step 5: Tokenization and Label Alignment Function
def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, padding='max_length', max_length=128)
    labels = []
    label_to_id = {label: idx for idx, label in enumerate(set(label for sublist in examples['labels'] for label in sublist))}

    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(tokenized_inputs['input_ids'][i])

        for j, label_id in enumerate(label):
            if j < len(word_ids) and word_ids[j] is not None:
                label_ids[word_ids[j]] = label_to_id[label_id]

        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Step 6: Model Fine-tuning Function
def fine_tune_model(model_name, train_dataset, val_dataset):
    if model_name == 'xlm-roberta-base':
        model = XLMRobertaForTokenClassification.from_pretrained(model_name, num_labels=len(set(label for _, labels in train_data for label in labels)))
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
    elif model_name == 'distilbert-base-uncased':
        model = DistilBertForTokenClassification.from_pretrained(model_name, num_labels=len(set(label for _, labels in train_data for label in labels)))
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    elif model_name == 'bert-base-multilingual-cased':
        model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(set(label for _, labels in train_data for label in labels)))
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

    train_tokenized = train_dataset.map(lambda x: tokenize_and_align_labels(tokenizer, x), batched=True)
    val_tokenized = val_dataset.map(lambda x: tokenize_and_align_labels(tokenizer, x), batched=True)

    training_args = TrainingArguments(
        output_dir=f"./results/{model_name.replace('/', '-')}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
    )

    trainer.train()

    return trainer, tokenizer

# Step 7: Evaluate Model Performance
def evaluate_model(trainer, val_dataset, tokenizer):
    outputs = trainer.predict(val_dataset)
    predictions = np.argmax(outputs.predictions, axis=2)

    true_labels = []
    preds = []

    for i, label_ids in enumerate(outputs.label_ids):
        true_labels.extend([label for label in label_ids if label != -100])
        preds.extend([predictions[i][j] for j, label in enumerate(label_ids) if label != -100])

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')
    accuracy = accuracy_score(true_labels, preds)

    return accuracy, precision, recall, f1

# Step 8: Fine-tune and Evaluate Multiple Models
model_names = ['xlm-roberta-base', 'distilbert-base-uncased', 'bert-base-multilingual-cased']
results = {}

for model_name in model_names:
    print(f"Fine-tuning {model_name}...")
    trainer, tokenizer = fine_tune_model(model_name, train_dataset, val_dataset)
    accuracy, precision, recall, f1 = evaluate_model(trainer, val_dataset, tokenizer)
    
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    print(f"Results for {model_name}: {results[model_name]}")

# Step 9: Comparative Analysis
print("\nComparative Analysis:")
for model_name, metrics in results.items():
    print(f"{model_name}: {metrics}")

# Step 10: Selection Criteria
best_model = max(results, key=lambda x: results[x]['f1_score'])
print(f"\nBest model based on F1-score: {best_model}")

# Step 11: Visualization of Results
def plot_results(results):
    models = list(results.keys())
    accuracy = [results[m]['accuracy'] for m in models]
    precision = [results[m]['precision'] for m in models]
    recall = [results[m]['recall'] for m in models]
    f1_scores = [results[m]['f1_score'] for m in models]

    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width * 1.5, accuracy, width, label='Accuracy')
    bars2 = ax.bar(x - width / 2, precision, width, label='Precision')
    bars3 = ax.bar(x + width / 2, recall, width, label='Recall')
    bars4 = ax.bar(x + width * 1.5, f1_scores, width, label='F1 Score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison for NER')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison_results.jpg', format='jpg')
    plt.show()

# Call the plot_results function
plot_results(results)
