import nemoguardrails
from nemoguardrails import LLMRails, RailsConfig
import pandas as pd
from tqdm import tqdm
import argparse
import os
import json
import sys

# Set up argument parser
parser = argparse.ArgumentParser(description="Process input CSV, config path, and output CSV.", fromfile_prefix_chars='@')
parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
parser.add_argument("--split_file", type=str, required=False, help="Path to the split file.")
parser.add_argument("--config_path", type=str, required=True, help="Path to the RailsConfig directory.")
parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file.")

args = parser.parse_args()
print(args)

# Load the input CSV
df = pd.read_csv(args.input_csv)
if args.split_file is not None:
    with open(args.split_file, "r") as f:
        split_indices = [int(line.strip()) for line in f.readlines()]
    df = df.iloc[split_indices]


# Load the RailsConfig
config = RailsConfig.from_path(args.config_path)

# Initialize the LLMRails app
app = LLMRails(config, verbose=True)
responses = []

# Define cache file path
cache_file = "cache.json"

# Load cache if it exists
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        cache = json.load(f)
else:
    cache = {}

# Process each row in the input CSV
for _, row in tqdm(df.iterrows(), total=len(df)):
    behavior_text = row["Behavior"]
    
    # Check if response is already cached
    if behavior_text in cache:
        new_message = cache[behavior_text]
    else:
        for attempt in range(3):  # Retry up to 3 times
            try:
                new_message = app.generate(messages=[{
                    "role": "user",
                    "content": behavior_text
                }])
            except nemoguardrails.actions.llm.utils.LLMCallException:
                print(f"LLM call failed for behavior: {behavior_text}. Attempt {attempt + 1}", file=sys.stderr)
                new_message = {"content": ""}
            if new_message["content"].strip():  # Check if the output is not an empty string
                break
        else:
            new_message = {"content": f"<<<ERROR>>> {app.explain().llm_calls[-1].completion}"}  # Fallback if all attempts fail
        cache[behavior_text] = new_message

    responses.append(new_message["content"])

    # Save responses to cache during processing
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

# Delete cache file after successful processing
if os.path.exists(cache_file):
    os.remove(cache_file)

# Add responses to the DataFrame and save to the output CSV
df["llm_response"] = responses
df.to_csv(args.output_csv, index=False)
