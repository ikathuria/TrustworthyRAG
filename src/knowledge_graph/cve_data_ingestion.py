from datetime import datetime, timedelta
from nvd_client import NvdApi


class CVEIngestion():
    """Class for ingesting CVE data into Neo4j"""

    def __init__(self, driver):
        self.driver = driver
        self.nvd_api = NvdApi()

    def fetch_cve_data(self, length=5, limit=1000):
        """Fetch CVE data from NVD API within the last 'length' days.
        
        Args:
            length (int): Number of days to look back for CVE data.
            limit (int): Maximum number of CVE records to fetch.
        
        Returns:
            dict: CVE data in JSON format.
        """
        cves_by_date = self.nvd_api.get_cve_by_date(
            per_page=100,
            offset=0,
            publish_start_date=datetime.now() - timedelta(days=length),
            publish_end_date=datetime.now()
        )
        return cves_by_date['vulnerabilities']


    def populate_nodes(self):
        """Populate Neo4j instance with CVE nodes.

        Args:
            cve_data (dict): CVE data in JSON format.

        Returns:
            None
        """

        cve_data = self.fetch_cve_data()

        with self.driver.session() as session:
            for cve in cve_data:
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
