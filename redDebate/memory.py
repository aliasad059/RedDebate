import os
import time
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
import hashlib


class Memory:
    """
    Base memory class providing common memory operations.

    Serves as the foundation for different memory types (short-term, long-term, vector-based)
    with basic storage, retrieval, and persistence functionality.

    Attributes:
        description (str): Description of the memory's purpose
        memories (list): List storing memory entries
    """
    def __init__(self,
                 description
                 ) -> None:
        self.description = description
        self.memories = []

    def clear(self) -> None:
        self.memories = []

    def add(self, memory: str, metadata: dict=None) -> None:
        self.memories.append(memory)

    def update(self, memory: str, index: int = -1) -> None:
        self.memories[index] = memory

    def save(self, file_path: str = None) -> None:
        with open(file_path, 'w') as f:
            f.write(f"{self.description}:\n")
            for memory in self.memories:
                f.write(f"{memory}\n")

    def load(self, file_path: str) -> None:
        with open(file_path, 'r') as f:
            self.description = f.readline().strip()
            for line in f:
                self.memories.append(line.strip())

    def __len__(self) -> int:
        return sum([len(memory.split()) for memory in self.memories])

class ShortTermMemory(Memory):
    """
    Short-term memory for storing recent conversation rounds and agent responses.

    Designed for temporary storage of recent interactions, used for
    maintaining context within a single debate session.
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
    """
    Basic Long-term memory for storage of key safety insights inside an array.
    """
    def __init__(self, description: str) -> None:
        super().__init__(description)

    def __str__(self) -> str:
        formatted_string = f"{self.description}: \n"

        if not self.memories:
            return formatted_string + "  No memories stored yet."

        for memory in self.memories:
            formatted_string += f"  {memory}\n"
        return formatted_string

class VectorStoreMemory(Memory):
    """
    Vector-based memory using Pinecone for semantic similarity search.

    Also known as Textual LTM in the paper. Provides a more advanced memory storage and retrieval
    using embedding-based similarity search for contextually relevant information.

    Attributes:
        pc (Pinecone): Pinecone client instance
        index: Pinecone index for vector storage
        embeddings (OpenAIEmbeddings): OpenAI embeddings model
        vector_store (PineconeVectorStore): LangChain Pinecone vector store
    """
    def __init__(self, description: str, index_name: str="red-debate-memory", emb_dimension: int=3072, similarity_metric: str="cosine") -> None:
        """
        Initialize vector store memory with Pinecone backend.

        Args:
            description (str): Memory description
            index_name (str): Pinecone index name
            emb_dimension (int): Embedding dimension (default: 3072 for text-embedding-3-large)
            similarity_metric (str): Similarity metric for vector search
        """
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

    def add(self, memory: str, metadata: dict=None) -> None:
        doc = Document(page_content=memory, metadata=metadata or {})
        doc_id = hashlib.sha256(memory.encode('utf-8')).hexdigest()

        try:
            self.vector_store.add_documents([doc], ids=[doc_id])
        except Exception as e:
            print(f"Error adding document to vector store: {e}")

    def retrieve(self, query: str, k: int = 5, filter: dict = None) -> tuple:
        results = self.vector_store.similarity_search(query, k=k, filter=filter)
        contents, metadata = [], []
        for result in results:
            contents.append(result.page_content)
            metadata.append(result.metadata)
        return contents, metadata

    def update_vector_memory(self, query: str, k: int = 5, filter: dict = None) -> None: # call this method to update the memory based on the new question asked
        contents, _ = self.retrieve(query, k=k, filter=filter)
        self.memories = contents

    def __str__(self) -> str:
        formatted_string = f"{self.description}: \n"

        if not self.memories:
            return formatted_string + "  No memories stored yet."

        for memory in self.memories:
            formatted_string += f"  {memory}\n"
        return formatted_string