# Hybrid Retriever with LangChain

This directory contains the implementation of a hybrid retriever system that leverages multiple retrieval techniques to enhance information retrieval capabilities. The system is designed to handle diverse query types and modalities, ensuring robust performance across various use cases.

The hybrid retriever integrates the following key components:
1. **Dense Retriever** - Utilizes dense vector representations to capture semantic similarities between queries and documents.
2. **Sparse Retriever** - Employs traditional keyword-based search techniques to retrieve relevant documents based on exact matches.
3. **Hybrid Fusion Module** - Combines the strengths of both dense and sparse retrievers, leveraging their respective advantages for improved retrieval performance.
