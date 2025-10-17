import requests
from src.data_ingestion.cybersec_data_ingestion import DataIngestion


class CVEIngestion(DataIngestion):
    """Class for ingesting CVE data into Neo4j"""

    def __init__(self, driver):
        super().__init__(driver)

    def fetch_cve_data(self, year=2024, limit=1000):
        """Fetch CVE data from NVD API.

        Args:
                year (int): Year for which to fetch CVE data.
                limit (int): Maximum number of CVE records to fetch.

        Returns:
                dict: CVE data in JSON format.
        """

        url = f"https://services.nvd.nist.gov/rest/json/cves/2.0"
        params = {
            'pubStartDate': f'{year}-01-01T00:00:00.000',
            'pubEndDate': f'{year}-12-31T23:59:59.999',
            'resultsPerPage': limit
        }
        response = requests.get(url, params=params)
        return response.json()

    def populate_cve_nodes(self, cve_data):
        """Populate Neo4j instance with CVE nodes.

        Args:
                cve_data (dict): CVE data in JSON format.

        Returns:
                None
        """

        with self.driver.session() as session:
            for cve in cve_data['vulnerabilities']:
                cve_info = cve['cve']
                session.run("""
					MERGE (c:CVE {id: $cve_id})
					SET c.description = $description,
						c.published_date = $pub_date,
						c.severity = $severity
				""",
                            cve_id=cve_info['id'],
                            description=cve_info.get('descriptions', [{}])[
                                0].get('value', ''),
                            pub_date=cve_info.get('published', ''),
                            severity=cve_info.get('metrics', {}).get('cvssMetricV3', [{}])[
                                0].get('cvssData', {}).get('baseScore', 0)
                            )
