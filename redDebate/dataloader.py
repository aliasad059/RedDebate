import pandas as pd
import os
import json
import random

class HarmBenchLoader:
    def __init__(self, dataset_path):
        self.dataset = self.load_dataset_harmbench(dataset_path)

    def load_dataset_harmbench(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
        dataset.rename(columns={'Behavior': 'text', 'SemanticCategory': 'label'}, inplace=True) # rename columns for consistency
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'text': self.dataset['text'].iloc[idx],
            'label': self.dataset['label'].iloc[idx]
        }

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self):
        return f'HarmBenchLoader: {len(self)} samples'


class CoSafeDatasetLoader:
    def __init__(self, dataset_path):
        self.dataset = self.load_dataset_cosafe(dataset_path)

    def load_dataset_cosafe(self, dataset_path):
        dataset = pd.read_csv(dataset_path) # use preprocess code to adopt data to the suitable format.
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'text': self.dataset['text'].iloc[idx],
            'label': self.dataset['label'].iloc[idx]
        }

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self):
        return f'CoSafeDatasetLoader: {len(self)} samples'

    def preprocess_conversations(self, file_path):
        formatted_conversations = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                conversations = json.loads(line)
                context = ""  # Collecting conversation context
                formatted_text = ""

                for message in conversations:
                    role = "User" if message["role"] == "user" else "Agent"
                    formatted_text += f"- **{role}:** {message['content']}\n"

                    # Append user messages for context tracking
                    if message["role"] == "user":
                        context += message["content"] + " "

                # Adding structured output
                structured_output = f"Here is the conversation so far between you and the user. Please take this into account and answer the next question.\n{formatted_text}\n---\n"
                formatted_conversations.append(structured_output)

        return formatted_conversations

    def preprocess_all_conversations(self, folder_path):
        all_data = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(folder_path, file_name)
                formatted_conversations = self.preprocess_conversations(file_path)

                # Create a label for each conversation, which is the file name without the .json extension
                label = file_name.replace(".json", "")

                for i, conversation in enumerate(formatted_conversations):
                    all_data.append({
                        "text": conversation,
                        "label": label,
                        "index": i,
                    })

        # Convert list of dictionaries to a DataFrame
        df = pd.DataFrame(all_data)

        # Shuffle the dataframe
        df = df.sample(frac=1).reset_index(drop=True)

        # Save to CSV
        df.to_csv("formatted_cosafe_v2.csv", index=False, encoding="utf-8")
        print("Formatted conversations saved successfully in 'formatted_cosafe.csv'!")

class TrivaQADatasetLoader:
    def __init__(self, dataset_path):
        self.dataset = self.load_dataset_trivia(dataset_path)

    def load_dataset_trivia(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'text': self.dataset['question'].iloc[idx],
            'label': self.dataset['question_id'].iloc[idx]
        }

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self):
        return f'TriviaQADatasetLoader: {len(self)} samples'

def load_datasets(dataset_names=None):
    datasets = {}

    if dataset_names is not None:
        print('Loading datasets...')
        for dataset in dataset_names:
            dataset_name, dataset_path = dataset.split(':')
            if dataset_name == 'harmbench':
                harmbench_loader = HarmBenchLoader(dataset_path)
                datasets['harmbench'] = harmbench_loader
            elif dataset_name == 'cosafe':
                cosafe_loader = CoSafeDatasetLoader(dataset_path)
                datasets['cosafe'] = cosafe_loader
            elif dataset_name == 'triviaqa':
                triviaqa_loader = TrivaQADatasetLoader(dataset_path)
                datasets['triviaqa'] = triviaqa_loader
            else:
                raise ValueError(f"Dataset '{dataset_name}' not supported. Please choose from 'harmbench', 'cosafe', 'toxicchat', 'hhrlhf', or 'saferdialogues'.")

    print(f'Loaded datasets: {list(datasets.keys())}')
    return datasets


if __name__ == '__main__':
    datasets = load_datasets()

    for key, loader in datasets.items():
        print(f'{key}: {len(loader)} samples')
        print(loader[0])
        print()