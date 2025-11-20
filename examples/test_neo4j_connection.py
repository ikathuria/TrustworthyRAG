"""
Quick test script to verify Neo4j connection.
Use this to check if your local Neo4j instance is accessible.
"""

from src.neo4j.neo4j_manager import Neo4jManager
import src.utils.constants as C


def test_connection():
    """Test Neo4j connection"""
    print("Testing Neo4j connection...")
    print(f"URI: {C.NEO4J_URI}")
    print(f"Database: {C.NEO4J_DB}")
    print(f"Username: {C.NEO4J_USERNAME}")
    print()
    
    try:
        neo4j_manager = Neo4jManager(
            uri=C.NEO4J_URI,
            username=C.NEO4J_USERNAME,
            password=C.NEO4J_PASSWORD,
            database=C.NEO4J_DB
        )
        
        print("✅ Connection successful!")
        
        # Test query
        result = neo4j_manager.query_graph("RETURN 1 as test")
        print(f"✅ Query test successful: {result}")
        
        # Get statistics
        stats = neo4j_manager.get_statistics()
        print("\n📊 Database Statistics:")
        print(f"   Total Entities: {stats.get('total_entities', 0)}")
        print(f"   Total Relations: {stats.get('total_relations', 0)}")
        
        neo4j_manager.close()
        print("\n✅ All tests passed! Neo4j is ready for QALF.")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Is Neo4j running? Check: neo4j status")
        print("  2. Are the connection settings correct in src/utils/constants.py?")
        print("  3. Is the password correct?")
        print("  4. Is Neo4j listening on the correct port (default: 7687)?")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_connection()

