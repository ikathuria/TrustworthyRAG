# Task 1: Knowledge Graph Construction - Implementation Guide
print("=== TASK 1: KNOWLEDGE GRAPH CONSTRUCTION ===")
print()

# Create detailed implementation guide for Task 1
task1_guide = """
# Task 1: Knowledge Graph Construction Implementation Guide

## Overview
Build a foundational cybersecurity knowledge graph integrating MITRE ATT&CK, CVE, CWE, and CAPEC data sources with Neo4j as the backend and LangChain/LlamaIndex for GraphRAG functionality.

## Step 1: Environment Setup

### Prerequisites
- Python 3.10+
- Neo4j Database (Community Edition)
- 16GB+ RAM recommended
- GPU optional but recommended for embeddings

### Installation Commands
```bash
# Create virtual environment
conda create -n cybergraphrag python=3.10
conda activate cybergraphrag

# Install core dependencies
pip install neo4j==5.14.1
pip install langchain==0.1.0
pip install langchain-community
pip install langchain-neo4j
pip install llamaindex==0.9.15
pip install stix2==3.0.1
pip install requests pandas numpy
pip install sentence-transformers
pip install openai  # or other LLM provider

# Install specialized tools
pip install ontolocy  # For MITRE ATT&CK parsing
```

## Step 2: Neo4j Database Setup

### 1. Install Neo4j Desktop
- Download from https://neo4j.com/download/
- Create new database project
- Set password and configure memory settings

### 2. Database Configuration
- Heap Memory: 8GB minimum
- Page Cache: 4GB minimum
- Enable APOC procedures for advanced operations

### 3. Connection Testing
```python
from neo4j import GraphDatabase

# Test connection
driver = GraphDatabase.driver("bolt://localhost:7687", 
                             auth=("neo4j", "your_password"))

def test_connection(driver):
    with driver.session() as session:
        result = session.run("RETURN 'Hello World' as message")
        print(result.single()["message"])

test_connection(driver)
driver.close()
```

## Step 3: Data Source Integration

### MITRE ATT&CK Integration
Use the Ontolocy parser for automated MITRE ATT&CK ingestion:

```python
# Using Ontolocy for MITRE ATT&CK
import os
os.environ['NEO4J_URI'] = 'neo4j://localhost:7687'
os.environ['NEO4J_USERNAME'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'your_password'

# Command line usage
# ontolocy parse mitre-attack
```

### CVE Data Integration
```python
import requests
import json
from neo4j import GraphDatabase

def fetch_cve_data(year=2024, limit=1000):
    \"\"\"Fetch CVE data from NVD API\"\"\"
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {
        'pubStartDate': f'{year}-01-01T00:00:00.000',
        'pubEndDate': f'{year}-12-31T23:59:59.999',
        'resultsPerPage': limit
    }
    response = requests.get(url, params=params)
    return response.json()

def populate_cve_nodes(driver, cve_data):
    \"\"\"Populate Neo4j with CVE nodes\"\"\"
    with driver.session() as session:
        for cve in cve_data['vulnerabilities']:
            cve_info = cve['cve']
            session.run(\"\"\"
                MERGE (c:CVE {id: $cve_id})
                SET c.description = $description,
                    c.published_date = $pub_date,
                    c.severity = $severity
            \"\"\", 
            cve_id=cve_info['id'],
            description=cve_info.get('descriptions', [{}])[0].get('value', ''),
            pub_date=cve_info.get('published', ''),
            severity=cve_info.get('metrics', {}).get('cvssMetricV3', [{}])[0].get('cvssData', {}).get('baseScore', 0)
            )
```

### CWE Data Integration
```python
def populate_cwe_data(driver):
    \"\"\"Populate CWE (Common Weakness Enumeration) data\"\"\"
    # Download CWE XML from https://cwe.mitre.org/data/xml/cwec_latest.xml
    # Parse and populate - implementation depends on XML structure
    
    sample_cwes = [
        {"id": "CWE-79", "name": "Cross-site Scripting", "description": "Improper Neutralization of Input"},
        {"id": "CWE-89", "name": "SQL Injection", "description": "Improper Neutralization of Special Elements"},
        {"id": "CWE-20", "name": "Improper Input Validation", "description": "Product does not validate input properly"}
    ]
    
    with driver.session() as session:
        for cwe in sample_cwes:
            session.run(\"\"\"
                MERGE (c:CWE {id: $cwe_id})
                SET c.name = $name,
                    c.description = $description
            \"\"\", cwe_id=cwe['id'], name=cwe['name'], description=cwe['description'])
```

### CAPEC Data Integration
```python
def populate_capec_data(driver):
    \"\"\"Populate CAPEC (Common Attack Pattern Enumeration) data\"\"\"
    sample_capecs = [
        {"id": "CAPEC-66", "name": "SQL Injection", "description": "Adversary injects SQL commands"},
        {"id": "CAPEC-85", "name": "AJAX Fingerprinting", "description": "Adversary explores web application"}
    ]
    
    with driver.session() as session:
        for capec in sample_capecs:
            session.run(\"\"\"
                MERGE (c:CAPEC {id: $capec_id})
                SET c.name = $name,
                    c.description = $description
            \"\"\", capec_id=capec['id'], name=capec['name'], description=capec['description'])
```

## Step 4: Relationship Mapping

### Define Ontology Schema
```python
def create_schema_constraints(driver):
    \"\"\"Create database constraints and indexes\"\"\"
    with driver.session() as session:
        # Unique constraints
        session.run("CREATE CONSTRAINT cve_id IF NOT EXISTS FOR (c:CVE) REQUIRE c.id IS UNIQUE")
        session.run("CREATE CONSTRAINT cwe_id IF NOT EXISTS FOR (c:CWE) REQUIRE c.id IS UNIQUE")
        session.run("CREATE CONSTRAINT capec_id IF NOT EXISTS FOR (c:CAPEC) REQUIRE c.id IS UNIQUE")
        
        # Indexes for performance
        session.run("CREATE INDEX cve_severity IF NOT EXISTS FOR (c:CVE) ON (c.severity)")
        session.run("CREATE INDEX cve_date IF NOT EXISTS FOR (c:CVE) ON (c.published_date)")

def create_relationships(driver):
    \"\"\"Create relationships between entities\"\"\"
    with driver.session() as session:
        # CVE to CWE relationships
        session.run(\"\"\"
            MATCH (cve:CVE), (cwe:CWE)
            WHERE cve.description CONTAINS 'injection' AND cwe.id = 'CWE-89'
            MERGE (cve)-[:EXPLOITS]->(cwe)
        \"\"\")
        
        # CWE to CAPEC relationships
        session.run(\"\"\"
            MATCH (cwe:CWE), (capec:CAPEC)
            WHERE cwe.id = 'CWE-89' AND capec.id = 'CAPEC-66'
            MERGE (cwe)-[:ENABLES]->(capec)
        \"\"\")
```

## Step 5: GraphRAG Implementation

### LangChain GraphRAG Setup
```python
from langchain.vectorstores import Neo4jVector
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph

class CybersecurityGraphRAG:
    def __init__(self, neo4j_url, username, password, openai_key):
        self.graph = Neo4jGraph(
            url=neo4j_url,
            username=username,
            password=password
        )
        self.embeddings = OpenAIEmbeddings(api_key=openai_key)
        self.qa_chain = GraphCypherQAChain.from_llm(
            llm=OpenAI(api_key=openai_key),
            graph=self.graph
        )
    
    def query(self, question: str):
        \"\"\"Query the cybersecurity knowledge graph\"\"\"
        return self.qa_chain.run(question)
    
    def add_vector_index(self, node_label: str, property_name: str):
        \"\"\"Add vector index for semantic search\"\"\"
        vector_store = Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=self.graph.url,
            username=self.graph.username,
            password=self.graph.password,
            node_label=node_label,
            text_node_properties=[property_name],
            embedding_node_property="embedding"
        )
        return vector_store

# Usage example
graph_rag = CybersecurityGraphRAG(
    neo4j_url="bolt://localhost:7687",
    username="neo4j", 
    password="your_password",
    openai_key="your_openai_key"
)

# Query examples
result1 = graph_rag.query("What CVEs are related to SQL injection?")
result2 = graph_rag.query("Show me attack patterns that exploit CWE-79")
```

## Step 6: Validation and Testing

### Data Quality Validation
```python
def validate_graph_structure(driver):
    \"\"\"Validate graph structure and data quality\"\"\"
    with driver.session() as session:
        # Check node counts
        counts = {}
        for label in ['CVE', 'CWE', 'CAPEC', 'Technique']:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
            counts[label] = result.single()['count']
        
        # Check relationship counts  
        rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
        counts['relationships'] = rel_result.single()['rel_count']
        
        print("Graph Statistics:", counts)
        return counts

def test_graph_queries(driver):
    \"\"\"Test basic graph queries\"\"\"
    with driver.session() as session:
        # Test multi-hop query
        result = session.run(\"\"\"
            MATCH (cve:CVE)-[:EXPLOITS]->(cwe:CWE)-[:ENABLES]->(capec:CAPEC)
            RETURN cve.id, cwe.id, capec.id
            LIMIT 10
        \"\"\")
        
        print("Multi-hop relationships:")
        for record in result:
            print(f"CVE {record['cve.id']} -> CWE {record['cwe.id']} -> CAPEC {record['capec.id']}")
```

## Expected Outputs

1. **Populated Neo4j Database** with 200K+ nodes and 500K+ relationships
2. **GraphRAG Query Interface** supporting natural language queries
3. **Data Ingestion Pipeline** for continuous updates
4. **Validation Reports** confirming data quality and completeness

## Timeline
- Week 1-2: Environment setup and data source integration
- Week 3-4: Ontology design and relationship mapping  
- Week 5-6: GraphRAG implementation and testing
- Week 7-8: Validation, optimization, and documentation

This implementation provides the foundation for advanced GraphRAG capabilities in cybersecurity contexts.
"""

# Save Task 1 guide
with open('task1_knowledge_graph_guide.md', 'w') as f:
    f.write(task1_guide)

print("✅ Task 1 Implementation Guide saved to 'task1_knowledge_graph_guide.md'")
print()
print("Key components covered:")
print("- Environment setup and dependencies")
print("- Neo4j database configuration")
print("- MITRE ATT&CK, CVE, CWE, CAPEC data integration")
print("- Relationship mapping and schema design")
print("- GraphRAG implementation with LangChain")
print("- Validation and testing procedures")
