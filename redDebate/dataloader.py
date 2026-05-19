"""Dataset loaders for safety-evaluation corpora.

Every loader exposes the same minimal indexable interface so that
:func:`load_datasets` can mix and match them by name on the CLI:

* ``__len__``                – number of samples.
* ``__getitem__(idx)``       – returns ``{"text": str, "label": str, ...}``.
* ``__iter__``               – yields the same dicts in order.

Supported dataset names (used as the prefix in ``--datasets``):
``harmbench``, ``cosafe``, ``aegis2``, ``triviaqa``,
``xstest``, ``dummy``.
"""

import pandas as pd
import os
import json


class HarmBenchLoader:
    """HarmBench behavior CSV loader.

    Renames ``Behavior`` → ``text`` and ``SemanticCategory`` → ``label``.
    """

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

class Aegis2DatasetLoader: # choose between train/validation/test
    """NVIDIA Aegis 2 safety dataset loader.

    Renames ``prompt`` → ``text`` and ``violated_categories`` → ``label``.
    """

    def __init__(self, dataset_path):
        self.dataset = self.load_dataset_aegis2(dataset_path)

    def load_dataset_aegis2(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
        dataset.rename(columns={'prompt': 'text', 'violated_categories': 'label'}, inplace=True) # rename columns for consistency

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
        return f'Aegis2DatasetLoader: {len(self)} samples'


class DummyLoader:
    """JSONL loader used for local smoke tests; supports a per-question
    ``memory`` field so :class:`~redDebate.memory.LongTermMemory` can be
    pre-seeded for evaluation."""

    def __init__(self, dataset_path):
        self.dataset = self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path):
        with open(dataset_path, "r") as f:
            dataset = [json.loads(line) for line in f]
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'text': self.dataset[idx]['text'],
            'label': self.dataset[idx]['label'],
            'memory': self.dataset[idx]['memory'],
        }

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self):
        return f'DummyLoader: {len(self)} samples'


class CoSafeDatasetLoader:
    """CoSafe pre-formatted multi-turn conversation CSV loader.

    Each JSONL line in the source files contains a list of ``{role, content}``
    messages representing a multi-turn conversation.

    Before loading into this dataset, conversations should be rendered into a
    Markdown-style transcript and appended with a final instruction asking the
    agent to continue the dialogue.

    Example preprocessing function::

        def process_conversations(file_path):
            formatted_conversations = []
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    conversations = json.loads(line)
                    context = ""
                    formatted_text = ""
                    for message in conversations:
                        role = (
                            "User"
                            if message["role"] == "user"
                            else "Agent"
                        )
                        formatted_text += (
                            f"- **{role}:** {message['content']}\n"
                        )
                        if message["role"] == "user":
                            context += message["content"] + " "
                    structured_output = (
                        "Here is the conversation so far between you and "
                        "the user. Please take this into account and answer "
                        "the next question.\n"
                        f"{formatted_text}\n---\n"
                    )
                    formatted_conversations.append(structured_output)
            return formatted_conversations
    """

    def __init__(self, dataset_path):
        self.dataset = self.load_dataset_cosafe(dataset_path)

    def load_dataset_cosafe(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
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


class TrivaQADatasetLoader:
    """TriviaQA loader. ``label`` is the ``question_id`` (used as a tag, not safety)."""

    def __init__(self, dataset_path):
        self.dataset = self.load_dataset_trivia(dataset_path)

    # path = '/home/xiaodan/Desktop/RedDebate/datasets/cosafe/CoSafe datasets/formatted_cosafe.csv'
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

class XSTestDatasetLoader:
    """XSTest loader. ``label`` is the prompt ``type`` (safe vs. unsafe)."""

    def __init__(self, dataset_path):
        self.dataset = self.load_dataset_xstest(dataset_path)

    def load_dataset_xstest(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'text': self.dataset['question'].iloc[idx],
            'label': self.dataset['type'].iloc[idx]
        }

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self):
        return f'XSTestDatasetLoader: {len(self)} samples'


def load_datasets(dataset_names=None):
    """Build a ``{name: loader}`` dict from CLI-style ``name:path`` entries.

    Args:
        dataset_names: Iterable of ``"<dataset_name>:<dataset_path>"`` strings,
            where ``dataset_name`` is one of the keys listed in the module
            docstring. ``None`` returns an empty dict.

    Raises:
        ValueError: If any entry references an unknown dataset name.
    """
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
            elif dataset_name == 'aegis2':
                aegis2_loader = Aegis2DatasetLoader(dataset_path)
                datasets['aegis2'] = aegis2_loader
            elif dataset_name == 'triviaqa':
                triviaqa_loader = TrivaQADatasetLoader(dataset_path)
                datasets['triviaqa'] = triviaqa_loader
            elif dataset_name == 'xstest':
                xstest_loader = XSTestDatasetLoader(dataset_path)
                datasets['xstest'] = xstest_loader
            elif dataset_name == 'dummy':
                dummy_loader = DummyLoader(dataset_path)
                datasets['dummy'] = dummy_loader
            else:
                raise ValueError(f"Dataset '{dataset_name}' not supported. Please choose from 'harmbench', 'cosafe', 'aegis2', 'toxicchat', 'hhrlhf', or 'saferdialogues'.")

    print(f'Loaded datasets: {list(datasets.keys())}')

    return datasets


if __name__ == '__main__':
    datasets = load_datasets()

    for key, loader in datasets.items():
        print(f'{key}: {len(loader)} samples')
        print(loader[0])
        print()