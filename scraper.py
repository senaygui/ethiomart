from telethon import TelegramClient
import csv
import os
from dotenv import load_dotenv
from telethon.errors import FloodWaitError, ChannelPrivateError
import asyncio

# Load environment variables
load_dotenv('.env')
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('phone')

# Function to scrape data from a single channel
async def scrape_channel(client, channel_username, writer, media_dir):
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title  # Extract the channel's title
        async for message in client.iter_messages(entity, limit=10000):
            media_path = None
            if message.media and hasattr(message.media, 'photo'):
                filename = f"{channel_username}_{message.id}.jpg"
                media_path = os.path.join(media_dir, filename)
                await client.download_media(message.media, media_path)
            
            writer.writerow([channel_title, channel_username, message.id, message.message, message.date, media_path])

    except FloodWaitError as e:
        print(f"Rate limit hit, waiting for {e.seconds} seconds.")
        await asyncio.sleep(e.seconds)
        await scrape_channel(client, channel_username, writer, media_dir)
    except ChannelPrivateError:
        print(f"Cannot access {channel_username}, it might be private.")
    except Exception as e:
        print(f"An error occurred while scraping {channel_username}: {e}")

# Initialize the Telegram client
client = TelegramClient('scraping_session', api_id, api_hash)

async def main():
    await client.start()
    media_dir = './data/photos'
    os.makedirs(media_dir, exist_ok=True)

    with open('./data/telegram_data.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])

        channels = [
            '@standardkitchen',  # Add more channels as needed
        ]
        
        for channel in channels:
            await scrape_channel(client, channel, writer, media_dir)
            print(f"Scraped data from {channel}")

with client:
    client.loop.run_until_complete(main())
