import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
auth = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

def check_rels():
    try:
        with GraphDatabase.driver(uri, auth=auth) as driver:
            with driver.session() as session:
                print("--- ALL RELATIONSHIP TYPES ---")
                rels = session.run("CALL db.relationshipTypes()")
                for rel in rels:
                    print(f"- {rel[0]}")
                
                print("\n--- SAMPLE RELATIONSHIPS ---")
                sample = session.run("MATCH (n)-[r]->(m) RETURN labels(n) as l1, type(r) as t, labels(m) as l2 LIMIT 10")
                for record in sample:
                    print(f"{record['l1']} -[:{record['t']}]-> {record['l2']}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_rels()
