#!/usr/bin/env python3
"""
Build (or rebuild) the ChromaDB vector store from data/biography.txt.

Usage:
    python scripts/build_vectors.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

import config
import rag

load_dotenv()


def main():
    print(f"Loading biography from {config.BIOGRAPHY_TXT}...")
    text = config.BIOGRAPHY_TXT.read_text(encoding="utf-8")

    print("Chunking...")
    chunks = rag.chunk_curated_lines(text)
    print(f"  → {len(chunks)} chunks")

    print(f"Embedding with {config.EMBEDDING_MODEL}...")
    oai_client = OpenAI()
    rag.embed_chunks(oai_client, chunks)
    print(f"  → {len(chunks)} embeddings generated")

    print(f"Storing in ChromaDB at {config.CHROMA_PATH}...")
    chroma_client = chromadb.PersistentClient(
        config.CHROMA_PATH,
        config.CHROMA_CLIENT_SETTINGS,
    )
    collection = rag.db_store_embeds(chroma_client, config.CHROMA_COLLECTION_NAME, chunks)
    print(f"  → Collection '{config.CHROMA_COLLECTION_NAME}' created with {collection.count()} entries")

    print("Done!")


if __name__ == "__main__":
    main()
