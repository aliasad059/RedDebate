"""Memory backends shared across debate agents.

Three flavors are provided:

* :class:`ShortTermMemory` – the chat history of the current debate (cleared
  between questions).
* :class:`LongTermMemory` – an array of textual feedback rules accumulated
  across debates, supplied verbatim to every agent.
* :class:`VectorStoreMemory` – the same idea, but stored in a Pinecone index
  and retrieved by semantic similarity to the current question.

Each class exposes a ``__str__`` that yields the prompt-ready textual form.
"""

import os
import time
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
import hashlib


class Memory:
    """Base in-memory store: a description plus an ordered list of entries.

    Subclasses override :meth:`__str__` to render the memory in a prompt-friendly
    way and may override :meth:`add`/:meth:`update` to persist to a backend.
    """

    def __init__(self,
                 description
                 ) -> None:
        self.description = description
        self.memories = []

    def clear(self) -> None:
        """Drop every stored memory."""
        self.memories = []

    def add(self, memory: str, metadata: dict=None) -> None:
        """Append a new memory entry. ``metadata`` is ignored by the base class."""
        self.memories.append(memory)

    def update(self, memory: str, index: int = -1) -> None:
        """Replace the entry at ``index`` (default: the most recent one)."""
        self.memories[index] = memory

    def save(self, file_path: str = None) -> None:
        """Write the description and one memory per line to ``file_path``."""
        with open(file_path, 'w') as f:
            f.write(f"{self.description}:\n")
            for memory in self.memories:
                f.write(f"{memory}\n")

    def load(self, file_path: str) -> None:
        """Inverse of :meth:`save`. Reads ``file_path`` produced by ``save``."""
        with open(file_path, 'r') as f:
            self.description = f.readline().strip()
            for line in f:
                self.memories.append(line.strip())

    def __len__(self) -> int:
        """Total word count across all stored memories (used for token budgeting)."""
        return sum([len(memory.split()) for memory in self.memories])

class ShortTermMemory(Memory):
    """Per-debate transcript: each entry is a ``{agent_name: response_dict}`` round.

    Used by :class:`~redDebate.debate.Debate` to share the running chat history
    among debate agents inside a single question.
    """

    def __init__(self, description: str) -> None:
        super().__init__(description)

    def __str__(self) -> str:
        formatted_string = f"{self.description}: \n"

        if not self.memories:
            return formatted_string + "  No memories stored yet."

        for round_num, entry in enumerate(self.memories, start=1):
            formatted_string += f"Round {round_num}:\n"
            for agent, response in entry.items():
                formatted_string += f"  {agent} response: ' {response['response']} '"
                if 'is_safe' in response:
                     formatted_string += f"'Evaluated as '{'safe' if response['is_safe'] else 'unsafe'}'.\n"
        return formatted_string

class LongTermMemory(Memory):
    """Cross-debate textual memory: a flat list of feedback rules.

    The entire list is rendered into every agent's prompt, so size grows
    linearly with the number of debates that produced feedback. For larger
    runs use :class:`VectorStoreMemory` instead.
    """

    def __init__(self, description: str) -> None:
        super().__init__(description)

    def set_memories(self, memories: list) -> None:
        """Replace the memory list wholesale (used to seed pre-defined memory)."""
        self.memories = memories

    def __str__(self) -> str:
        formatted_string = f"{self.description}: \n"

        if not self.memories:
            return formatted_string + "  No memories stored yet."

        for memory in self.memories:
            formatted_string += f"  {memory}\n"
        return formatted_string

class VectorStoreMemory(Memory):
    """Pinecone-backed long-term memory with semantic retrieval.

    Each feedback entry is embedded with OpenAI ``text-embedding-3-large`` and
    stored in a serverless Pinecone index. Before each debate the caller
    invokes :meth:`update_vector_memory` with the current question to refresh
    ``self.memories`` with the ``k`` most-relevant prior feedbacks.

    Requires ``PINECONE_API_KEY`` and ``OPENAI_API_KEY`` in the environment.

    Args:
        description: Human-readable description rendered as the memory header.
        index_name: Pinecone index name (auto-created if missing).
        emb_dimension: Embedding dimension; must match the embedder.
        similarity_metric: Pinecone similarity metric, e.g. ``"cosine"``.
    """

    def __init__(self, description: str, index_name: str="red-debate-memory", emb_dimension: int=3072, similarity_metric: str="cosine") -> None:
        super().__init__(description)

        # initialize Pinecone
        self.pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=emb_dimension,
                metric= similarity_metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
        self.index = self.pc.Index(index_name)

        # initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # initialize Pinecone vector store
        self.vector_store = PineconeVectorStore(self.index, self.embeddings)

    def add(self, memory: str, metadata: dict=None) -> None: #TODO: do not add very similar feedbacks that are already in the memory
        """Embed and upsert ``memory`` into the index.

        The document id is the SHA-256 of the text so identical feedbacks are
        deduplicated by the upsert. Failures are caught and logged to stdout
        rather than raised so a single bad write doesn't kill a long run.
        """
        doc = Document(page_content=memory, metadata=metadata or {})
        doc_id = hashlib.sha256(memory.encode('utf-8')).hexdigest()

        try:
            self.vector_store.add_documents([doc], ids=[doc_id])
        except Exception as e:
            print(f"Error adding document to vector store: {e}")

    def retrieve(self, query: str, k: int = 5, filter: dict = None) -> tuple:
        """Return ``(contents, metadata)`` for the top-``k`` matches of ``query``."""
        results = self.vector_store.similarity_search(query, k=k, filter=filter)
        contents, metadata = [], []
        for result in results:
            contents.append(result.page_content)
            metadata.append(result.metadata)
        return contents, metadata

    def update_vector_memory(self, query: str, k: int = 5, filter: dict = None) -> None: # call this method to update the memory based on the new question asked
        """Refresh ``self.memories`` with the top-``k`` matches for ``query``.

        Call this once per question before the debate starts so that the
        prompt-rendered memory is question-specific.
        """
        contents, _ = self.retrieve(query, k=k, filter=filter)
        self.memories = contents

    def __str__(self) -> str:
        formatted_string = f"{self.description}: \n"

        if not self.memories:
            return formatted_string + "  No memories stored yet."

        for memory in self.memories:
            formatted_string += f"  {memory}\n"
        return formatted_string