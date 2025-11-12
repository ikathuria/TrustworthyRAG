# Database with Neo4j

We use Neo4j as our graph database to store and manage the structured data extracted from documents. Neo4j provides a powerful and flexible way to represent complex relationships between entities, making it ideal for our use case.

## Features of Neo4j
- **Graph Model**: Represents data as nodes, relationships, and properties.
- **Vector and Graph Data Support**: Efficiently handles both vector data and graph structures.
- **Scalability**: Capable of handling large datasets with complex relationships.
- **Integration**: Easily integrates with various programming languages and tools.

## Script Usage
To interact with the Neo4j database, we have created a class `Neo4jManager` in our ingestion script. We also have two other modules to ingestion the data into the graph  database and one to create vector indices. This is done to create a hybrid vector-graph database.
