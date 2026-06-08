from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from medical_agent.config import Settings, get_settings
from medical_agent.rag.bm25_store import BM25Store


@dataclass
class ChunkedDoc:
    chunk_id: str
    text: str
    source: str
    metadata: dict


def build_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def get_embed_model(settings: Settings) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=settings.embed_model, embed_batch_size=12)


def load_and_chunk_docs(docs_path: str, settings: Settings) -> list[ChunkedDoc]:
    reader = SimpleDirectoryReader(docs_path)
    documents: list[Document] = reader.load_data()

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
        paragraph_separator="\n\n",
    )
    nodes: list[TextNode] = splitter.get_nodes_from_documents(documents)

    chunks = []
    for node in nodes:
        source = Path(node.metadata.get("file_name", "unknown")).stem
        chunks.append(
            ChunkedDoc(
                chunk_id=node.node_id,
                text=node.text,
                source=source,
                metadata={**node.metadata, "source": source},
            )
        )
    return chunks


def build_qdrant_index(
    chunks: list[ChunkedDoc],
    client: QdrantClient,
    settings: Settings,
) -> VectorStoreIndex:
    collection = settings.qdrant_collection_rag

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    embed_model = get_embed_model(settings)
    vector_store = QdrantVectorStore(client=client, collection_name=collection)

    docs = [
        Document(text=c.text, metadata=c.metadata, id_=c.chunk_id) for c in chunks
    ]
    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=embed_model,
        vector_store=vector_store,
        show_progress=True,
    )
    return index


def populate_bm25(chunks: list[ChunkedDoc], bm25: BM25Store) -> None:
    for chunk in chunks:
        bm25.add(chunk.chunk_id, chunk.text)


def get_vector_store_index(
    client: QdrantClient,
    settings: Settings,
) -> VectorStoreIndex:
    embed_model = get_embed_model(settings)
    vector_store = QdrantVectorStore(
        client=client, collection_name=settings.qdrant_collection_rag
    )
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )


def run_ingest(settings: Settings | None = None) -> int:
    if settings is None:
        settings = get_settings()

    client = build_qdrant_client(settings)
    chunks = load_and_chunk_docs(settings.raw_docs_path, settings)

    build_qdrant_index(chunks, client, settings)

    bm25 = BM25Store()
    populate_bm25(chunks, bm25)
    bm25.save(settings.bm25_index_path)

    return len(chunks)
