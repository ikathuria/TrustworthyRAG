# Knowledge Graph Construction and Data Loading

This module is responsible for constructing a knowledge graph from various cybersecurity data sources and loading the data for use in the RAG pipeline. It includes functions to ingest data from CVE, CWE, and other relevant datasets, preprocess the data, and build a structured knowledge graph that can be queried by the retriever component of the RAG system.

## File Structure
- `cybersec_data_ingestion.py`: Main module for cybersecurity data ingestion.
- `cwe_data_ingestion.py`: Module for ingesting CWE data.
- `cve_data_ingestion.py`: Module for ingesting CVE data.
- `kg_construction.py`: Module for knowledge graph construction.

## Datasets
The data ingestion module utilizes the following datasets:

| Dataset Name | Description | Source | Modality |
|--------------|-------------|--------|----------|
| CVE (Common Vulnerabilities and Exposures) | A list of publicly disclosed cybersecurity vulnerabilities. | [NVD API](https://nvd.nist.gov/developers/vulnerabilities) | Text |
| CWE (Common Weakness Enumeration) | A list of common software and hardware weaknesses. | | [cwe2 package](https://github.com/aboutcode-org/cwe2) | Text |
| MITRE ATT&CK | A knowledge base of adversary tactics and techniques based on real-world observations. | [MITRE ATT&CK website](https://attack.mitre.org/) | Text |
| CAPEC (Common Attack Pattern Enumeration and Classification) | A comprehensive dictionary of known attack patterns. | [MITRE CAPEC website](https://capec.mitre.org/) | Text |
| APT notes | Detailed reports on Advanced Persistent Threats (APTs). | Various cybersecurity research publications. | Image |


### **CVE (Common Vulnerabilities and Exposures)**
A list of publicly disclosed cybersecurity vulnerabilities. We fetch real-time CVE data using requests module to access the [NVD (National Vulnerability Database) API](https://nvd.nist.gov/developers/vulnerabilities).

### **CWE (Common Weakness Enumeration)**
A list of common software and hardware weaknesses. The [cwe2 package](https://github.com/aboutcode-org/cwe2) is used for fetching and processing CWE data.

### MITRE ATT&CK
A knowledge base of adversary tactics and techniques based on real-world observations. Data can be obtained from the [MITRE ATT&CK website](https://attack.mitre.org/). We used 

