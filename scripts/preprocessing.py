import re
import pandas as pd

def extract_amharic_text_and_numbers_with_linebreaks(text):
    """
    Extracts Amharic text and numbers from a given string while preserving line breaks.
    :param text: The input string containing mixed language content.
    :return: A string containing Amharic characters and numbers with line breaks.
    """
    # Pattern to match Amharic characters and numbers (both Eastern and Western Arabic numerals)
    pattern = r'[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF0-9\u0660-\u0669\u1369-\u137C]+'
    
    # Replace non-matching characters (except line breaks) with spaces
    cleaned_text = re.sub(r'[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF0-9\u0660-\u0669\u1369-\u137C\n]+', ' ', text)
    return cleaned_text

def clean_dataframe_with_linebreaks(input_path, output_path):
    """
    Cleans the CSV file by extracting Amharic text and numbers from the 'Message' column,
    preserving line breaks, and saving the cleaned data.
    :param input_path: The path to the input CSV file.
    :param output_path: The path to save the cleaned CSV file.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(input_path)

    # Drop rows with NaN in critical columns like 'Message'
    df.dropna(subset=['Message'], inplace=True)

    # Apply the extract_amharic_text_and_numbers_with_linebreaks function to the 'Message' column
    df['Message'] = df['Message'].apply(lambda x: extract_amharic_text_and_numbers_with_linebreaks(str(x)))

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(output_path, index=False, encoding='utf-8')

def label_entities_conll_with_linebreaks(df, message_col='Message', output_file='../data/labeled_data.conll'):
    """
    Labels entities (Product, Price, and Location) in the dataset's messages column using the CoNLL format.
    Preserves line breaks and handles product labeling on the first line.
    :param df (pd.DataFrame): DataFrame containing the dataset with a column of messages.
    :param message_col (str): Name of the column containing the messages to be labeled.
    :param output_file (str): Output file to save the labeled data in CoNLL format.
    """
    # Static hardcoded location labels based on the provided example
    static_location_labels = [
        ("1", "O"), ("ፒያሳ", "B-LOC"), ("ጣይቱ", "I-LOC"), ("ሆቴል", "I-LOC"), ("ጊቢ", "I-LOC"),
        ("ውስጥ", "I-LOC"), ("ቢሮ", "I-LOC"), ("ቁ04", "I-LOC"), ("2", "O"), ("መገናኛ", "B-LOC"),
        ("መተባበር", "I-LOC"), ("ሕንፃ", "I-LOC"), ("3ኛ", "I-LOC"), ("ፎቅ", "I-LOC"),
        ("ቢሮ", "I-LOC"), ("ቁ316", "I-LOC"), ("3", "O"), ("ሳር", "B-LOC"), ("ቤት", "I-LOC"),
        ("ካናዳ", "I-LOC"), ("ኤምባሲ", "I-LOC"), ("ፊትለፊት", "I-LOC"), ("ሸዋ", "I-LOC"),
        ("ሱፕር", "I-LOC"), ("ማርኬት", "I-LOC"), ("ያለበት", "I-LOC"), ("2ኛ", "I-LOC"),
        ("ፎቅ", "I-LOC"), ("ቢር", "I-LOC"), ("ቁ11", "I-LOC"), ("4", "O"), ("ጀሞ", "B-LOC"),
        ("መስታወት", "I-LOC"), ("ፍብሪካ", "I-LOC"), ("ፊት", "I-LOC"), ("ለፊት", "I-LOC"),
        ("ከፍደም", "I-LOC"), ("ሞል", "I-LOC"), ("ግራውድ", "I-LOC"), ("ላይ", "I-LOC"),
        ("የሱቅ", "I-LOC"), ("ቁጥር", "I-LOC"), ("39", "I-LOC")
    ]

    def label_message_with_linebreaks(message):
        """
        Labels entities (Product, Price, and Location) in a message using custom logic or static locations.
        Preserves line breaks and labels the first line as a product.
        """
        lines = message.split('\n')  # Split the message into lines
        labeled_tokens = []
        
        for i, line in enumerate(lines):
            tokens = re.findall(r'\b\w+\b', line)  # Tokenize each line
            product_detected = False
            is_price = False
            is_location = False
            
            for token in tokens:
                # Match static locations
                for static_token, static_label in static_location_labels:
                    if token == static_token:
                        labeled_tokens.append((token, static_label))
                        break
                else:
                    # Apply dynamic rules for prices
                    if re.search(r'ዋጋ|ብር', token) or is_price:
                        if not is_price:
                            labeled_tokens.append((token, 'B-PRICE'))
                            is_price = True
                        else:
                            labeled_tokens.append((token, 'I-PRICE'))
                            if re.search(r'ብር$', token):
                                is_price = False
                    # Label the first line as product
                    elif i == 0 and not product_detected:
                        labeled_tokens.append((token, 'B-Product'))
                        product_detected = True
                    elif i == 0 and product_detected:
                        labeled_tokens.append((token, 'I-Product'))
                    else:
                        # Everything else is O
                        labeled_tokens.append((token, 'O'))
            
            # Add a newline to separate lines in CoNLL format
            labeled_tokens.append(("", "O"))
        
        return labeled_tokens

    # Write labeled data to the output file in CoNLL format
    with open(output_file, 'w', encoding='utf-8') as file:
        for message in df[message_col]:
            labeled_tokens = label_message_with_linebreaks(message)
            for token, label in labeled_tokens:
                if token:  # Write only if token is not an empty string (skip empty lines)
                    file.write(f"{token} {label}\n")
                else:
                    file.write("\n")  # Write a newline to separate lines
            file.write("\n")  # Separate messages



# Function to read a CoNLL formatted dataset into a pandas DataFrame
def load_conll_dataset(filepath):
    sentences = []
    labels = []
    sentence = []
    label = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == "":  # Empty lines separate sentences
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []
            else:
                token, tag = line.strip().split()  # assuming space-separated values
                sentence.append(token)
                label.append(tag)

    # Convert sentences and labels into a DataFrame
    df = pd.DataFrame({"tokens": sentences, "labels": labels})
    return df
