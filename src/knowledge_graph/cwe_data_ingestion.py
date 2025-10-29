from cwe2.database import Database
from src.data_ingestion.kg_construction import DataIngestion


class CWEIngestion(DataIngestion):
    """Class for ingesting CWE data into Neo4j"""

    def __init__(self, driver):
        super().__init__(driver)
        self.db = Database()

    def fetch_cwe_data(self):
        """Fetch CWE data using cwe2 library
        """

        return self.db.get_top_25_cwe()
    
    def print_sample_cwe_data(self, n=1):
        """Print sample CWE data for verification
        """

        sample_cwe = self.db.get(n=n)
        print(
            f"CWE ID: {sample_cwe['id']},\n"
            f"Name: {sample_cwe['name']},\n"
            f"Description: {sample_cwe['description']}\n"
        )

    def populate_nodes(self):
        """Populate CWE (Common Weakness Enumeration) data
        
        Returns:
            None
        """

        cwe_data = self.fetch_cwe_data()

        with self.driver.session() as session:
            for cwe in cwe_data:
                session.run("""
                            MERGE (c:CWE {id: $cwe_id})
                            SET c.name = $name,
                            c.description = $description
                            """,
                            cwe_id=cwe['id'],
                            name=cwe['name'],
                            description=cwe['description']
                            )


