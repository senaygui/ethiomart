import telethon
import pandas as pd
import re
import matplotlib.pyplot as plt
from telethon import TelegramClient

# Use your actual API ID and API Hash
api_id = '26737733'
api_hash = 'f590cc7e473a4e1c9ea4f7bc59163016'
client = TelegramClient('session_name', api_id, api_hash)

# List of Telegram channels to scrape
channels = [
    '@ZemenExpress',
    '@nevacomputer',
    '@meneshayeofficial',
    '@ethio_brand_collection',
    '@Leyueqa',
    '@sinayelj',
    '@Shewabrand',
    '@helloomarketethiopia',
    '@modernshoppingcenter',
    '@qnashcom',
    '@Fashiontera',
    '@kuruwear',
    '@gebeyaadama',
    '@MerttEka',
    '@forfreemarket'
]

async def fetch_messages(channel, limit=5000):  # Set default limit to 5000
    await client.start()
    messages = await client.get_messages(channel, limit=limit)
    return messages

def is_amharic(text):
    return any('\u1200' <= char <= '\u137F' for char in text)

def preprocess_text(text):
    text = re.sub(r'[^፩-፴መ-ዯ\s]', '', text)
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    token_count = len(tokens)
    return tokens, token_count

def save_preprocessed_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for metadata, tokens in data:
            record = f"Sender: {metadata['sender']}, Timestamp: {metadata['timestamp']}, Tokens: {', '.join(tokens)}\n"
            f.write(record)

async def main():
    all_data = []
    message_limit = 5000  # Set the limit to 5000 for each channel
    for channel in channels:
        messages = await fetch_messages(channel, limit=message_limit)
        for message in messages:
            if message.text and is_amharic(message.text):
                preprocessed_content, token_count = preprocess_text(message.text)
                metadata = {
                    'sender': message.sender_id,
                    'timestamp': message.date,
                    'token_count': token_count
                }
                all_data.append((metadata, preprocessed_content))
    save_preprocessed_data(all_data, 'preprocessed_data_all_channels.txt')

def label_data_for_conll(messages):
    labeled_data = []
    for msg in messages:
        tokens = msg[1]
        for token in tokens:
            entity_label = 'O'
            if re.match(r'^[\d]+$', token):
                entity_label = 'B-Price'
            elif re.match(r'^[፩-፴መ-ዯ\s]+$', token):
                entity_label = 'B-Product' if 'ምርት' in token else entity_label
            elif token in ['አዲስ አበባ', 'ዳር ዳዋ']:
                entity_label = 'B-Location'

            labeled_data.append(f"{token}\t{entity_label}")

        labeled_data.append("")  # Separate messages with a blank line

    with open('labeled_data_all_channels.conll', 'w', encoding='utf-8') as f:
        f.write('\n'.join(labeled_data))

def summarize_labeled_data(filename):
    entity_counts = {'B-Product': 0, 'B-Price': 0, 'B-Location': 0, 'O': 0}
    total_tokens = 0
    total_messages = 0
    tokens_per_message = []

    with open(filename, 'r', encoding='utf-8') as f:
        current_message_tokens = 0
        for line in f:
            if line.strip():
                token, label = line.rsplit('\t', 1)
                total_tokens += 1
                current_message_tokens += 1
                if label in entity_counts:
                    entity_counts[label] += 1
            else:
                if current_message_tokens > 0:
                    tokens_per_message.append(current_message_tokens)
                    total_messages += 1
                current_message_tokens = 0

    # Handle last message if the file does not end with a newline
    if current_message_tokens > 0:
        tokens_per_message.append(current_message_tokens)
        total_messages += 1

    average_tokens = total_tokens / total_messages if total_messages > 0 else 0

    # Create a statistics report
    plt.figure(figsize=(12, 8))

    # Entity counts bar chart
    plt.subplot(311)
    plt.bar(entity_counts.keys(), entity_counts.values(), color=['blue', 'green', 'orange', 'gray'])
    plt.title('Entity Count Summary')
    plt.xlabel('Entity Types')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Total messages and tokens
    plt.subplot(312)
    plt.bar(['Total Messages', 'Total Tokens'], [total_messages, total_tokens], color=['purple', 'red'])
    plt.title('Total Messages and Tokens')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Distribution of tokens per message
    plt.subplot(313)
    plt.hist(tokens_per_message, bins=range(1, max(tokens_per_message) + 2), color='skyblue', edgecolor='black')
    plt.title('Distribution of Tokens per Message')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Number of Messages')
    plt.grid(axis='y')

    # Save the figure as a JPG file
    plt.tight_layout()
    plt.savefig('statistics_report.jpg')
    plt.close()

if __name__ == "__main__":
    with client:
        client.loop.run_until_complete(main())

    # Load the preprocessed data to label a subset
    with open('preprocessed_data_all_channels.txt', 'r', encoding='utf-8') as f:
        subset_data = f.readlines()

    subset_messages = []
    for line in subset_data:
        if line.strip():
            parts = line.strip().split(', Tokens: ')
            metadata_part = parts[0].replace('Sender: ', '').replace('Timestamp: ', '').split(', ')
            tokens = parts[1].split(', ') if len(parts) > 1 else []
            subset_messages.append(({'sender': metadata_part[0], 'timestamp': metadata_part[1]}, tokens))

    label_data_for_conll(subset_messages)

    summarize_labeled_data('labeled_data_all_channels.conll')
