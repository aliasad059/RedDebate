import nemoguardrails
from nemoguardrails import LLMRails, RailsConfig
import pandas as pd
from tqdm import tqdm
import argparse
import os
import json
import sys

class Evaluator():
    def __init__(self):
        self.conversation_history = None
    
    def set_history(self, conversation_history):
        self.conversation_history = conversation_history

    def get_history(self) -> str:
        out = ""
        for turn in self.conversation_history:
            agent_str = "user" if turn["role"] == "user" else "bot"
            agent_verb = "said" if agent_str == "user" else "say"
            out += f"{agent_str} action: {agent_str} {agent_verb} \"{turn["content"]}\"\n"
        return out

# Set up argument parser
parser = argparse.ArgumentParser(description="Evalute on CoSafe.", fromfile_prefix_chars='@')
parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
parser.add_argument("--split_file", type=str, required=False, help="Path to the split file.")
parser.add_argument("--config_path", type=str, required=True, help="Path to the RailsConfig directory.")
parser.add_argument("--output_json", type=str, required=True, help="Path to save the output JSON file.")
parser.add_argument("--limit", type=int, default=-1, help="Limit the number of rows to process.")

args = parser.parse_args()
print(args)

# Load the input CSV
with open(args.input_json, "r") as f:
    data = json.load(f)
if args.split_file is not None:
    with open(args.split_file, "r") as f:
        split_indices = set([int(line.strip()) for line in f.readlines()])
    data = [data[i] for i in split_indices]
if args.limit > 0:
    data = data[:args.limit]

# Load the RailsConfig
config = RailsConfig.from_path(args.config_path)
evaluator = Evaluator()
config.evaluator = evaluator

# Initialize the LLMRails app
app = LLMRails(config, verbose=True)

# Define cache file path
cache_file = "cosafe_cache.json"

# Load cache if it exists
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        data = json.load(f)

# Process each row in the input CSV
for row in tqdm(data):
    if "llm_response" in row:  # Skip rows that already have a response
        continue

    conversation = row["conversation"]
    
    # TODO: retry up to 3 times if output is empty

    for attempt in range(3):  # Retry up to 3 times
        try:
            evaluator.set_history(conversation)
            new_message = app.generate(messages=[{
                "role": "user",
                "content": conversation[-1]["content"]
            }])
        except nemoguardrails.actions.llm.utils.LLMCallException as e:
            print(f"LLM call failed for input: {conversation[-1]["content"]}. Attempt {attempt + 1}", file=sys.stderr)
            print(e, file=sys.stderr)
            new_message = {"content": ""}
        if new_message["content"].strip():  # Check if the output is not an empty string
            break
    else:
        new_message = {"content": f"<<<ERROR>>> {app.explain().llm_calls[-1].completion}"}  # Fallback if all attempts fail

    row["llm_response"] = new_message["content"]

    # Save responses to cache during processing
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)

# Delete cache file after successful processing
if os.path.exists(cache_file):
    os.remove(cache_file)

# Save the output JSON
with open(args.output_json, "w") as f:
    json.dump(data, f, indent=2)
