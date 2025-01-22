import telethon
import pandas as pd
import re
import os
from telethon import TelegramClient
import asyncio

# Use your actual API ID and API Hash
api_id = '26737733'
api_hash = 'f590cc7e473a4e1c9ea4f7bc59163016'
client = TelegramClient('session_name', api_id, api_hash)

# List of Telegram channels to scrape
channels = [
    '@Leyueqa', '@sinayelj'
]

async def fetch_messages(channel):
    try:
        await client.start()
        messages = await client.get_messages(channel, limit=100)
        return messages
    except Exception as e:
        print(f"Error fetching messages from {channel}: {e}")
        return []

def is_amharic(text):
    # Check for Amharic characters (Unicode range)
    return any('\u1200' <= char <= '\u137F' for char in text)

def preprocess_text(text):
    # Normalize: removing non-Amharic characters
    text = re.sub(r'[^፩-፴መ-ዯ\s]', '', text)  # Keep only Amharic characters and spaces
    text = text.lower()  # Convert to lowercase

    # Tokenization: split into tokens
    tokens = re.findall(r'\w+', text)  # Basic tokenization
    return tokens

def save_preprocessed_data_as_txt(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for metadata, tokens in data:
            # Format: sender, timestamp, tokens separated by spaces
            f.write(f"{metadata['sender']}\t{metadata['timestamp']}\t" + "\t".join(tokens) + "\n")

async def main():
    all_data = []

    async def fetch_and_process(channel):
        messages = await fetch_messages(channel)
        for message in messages:
            if message.text and is_amharic(message.text):
                preprocessed_content = preprocess_text(message.text)
                metadata = {
                    'sender': message.sender_id,
                    'timestamp': message.date,
                }
                all_data.append((metadata, preprocessed_content))

    # Run fetch_and_process for all channels concurrently
    await asyncio.gather(*(fetch_and_process(channel) for channel in channels))

    # Save the preprocessed data as a text file
    save_preprocessed_data_as_txt(all_data, r'C:\Users\User\Desktop\10Acadamy\Week-Five\preprocessed_data.txt')

if __name__ == "__main__":
    with client:
        client.loop.run_until_complete(main())

