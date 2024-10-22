import pandas as pd
import argparse
import re

# Load your DataFrame
df = pd.read_csv('~/Dev/pythonProjects/train.csv')

# Set display options to show full content
pd.set_option('display.max_colwidth', None)

# Set up argument parsing
parser = argparse.ArgumentParser(description="Input desired essay IDs")
parser.add_argument('id1', type=str, help='The first ID to lookup')
parser.add_argument('id2', type=str, help='The second ID to lookup')
args = parser.parse_args()

# Perform the lookup
result1 = df.loc[df['essay_id'] == args.id1, 'full_text']
result2 = df.loc[df['essay_id'] == args.id2, 'full_text']

# Function to clean up text
def clean_text(text):
    if isinstance(text, str):
        # Replace line breaks with a space and strip leading/trailing whitespace
        cleaned_text = re.sub(r'\s+', ' ', text.replace('\\n', ' ')).strip()
        return cleaned_text
    return text

# Print the results, applying the cleaning function
print(result1.apply(clean_text).to_string(index=False))
print("********************************")
print(result2.apply(clean_text).to_string(index=False))

