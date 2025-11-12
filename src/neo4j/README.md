# Database with Neo4j, Mistral with Ollama

We use Neo4j as our graph database to store and manage the structured data extracted from documents. Neo4j provides a powerful and flexible way to represent complex relationships between entities, making it ideal for our use case. The Neo4j graph database is used to create a hybrid vector-graph database. This allows us to leverage the strengths of both graph databases and vector databases for efficient data retrieval and management.

We are also using Mistral to help extract entities and relationships from unstructured text data, which are then ingested into the Neo4j database. The script also allows customization of which LLM to use.

## Features of Neo4j
- **Graph Model**: Represents data as nodes, relationships, and properties.
- **Vector and Graph Data Support**: Efficiently handles both vector data and graph structures.
- **Scalability**: Capable of handling large datasets with complex relationships.
- **Integration**: Easily integrates with various programming languages and tools.

## Script Usage
To interact with the Neo4j database, we have created a class `Neo4jManager` in our ingestion script. We also have two other modules to ingestion the data into the graph  database and one to create vector indices. This is done to create a hybrid vector-graph database.
