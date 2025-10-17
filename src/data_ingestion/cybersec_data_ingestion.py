import os
import requests

os.environ['NEO4J_URI'] = 'neo4j://127.0.0.1:7687'
os.environ['NEO4J_USERNAME'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'test1234'


class DataIngestion:
    """Class for ingesting cybersecurity data into Neo4j"""

    def __init__(self, driver):
        self.driver = driver

