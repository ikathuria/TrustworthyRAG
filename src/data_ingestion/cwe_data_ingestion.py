from cwe2.database import Database
from src.data_ingestion.cybersec_data_ingestion import DataIngestion


class CWEIngestion(DataIngestion):
    """Class for ingesting CWE data into Neo4j"""

    def __init__(self, driver):
        super().__init__(driver)

    def fetch_cwe_data(self):
        """Fetch CWE data using cwe2 library
        """
        db = Database()
        return db.get_cwe_data()

    def populate_cwe_data(self):
        """Populate CWE (Common Weakness Enumeration) data"""
        # Download CWE XML from https://cwe.mitre.org/data/xml/cwec_latest.xml.zip
        # Parse and populate - implementation depends on XML structure

        cwe_data = self.fetch_cwe_data()

        with self.driver.session() as session:
            for cwe in cwe_data:
                session.run("""
                                        MERGE (c:CWE {id: $cwe_id})
                                        SET c.name = $name,
                                                c.description = $description
                                """, cwe_id=cwe['id'], name=cwe['name'], description=cwe['description'])


