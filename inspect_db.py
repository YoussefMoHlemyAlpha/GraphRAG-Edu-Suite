import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
auth = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

def inspect_labels_and_rels():
    try:
        with GraphDatabase.driver(uri, auth=auth) as driver:
            with driver.session() as session:
                print("--- Node Labels ---")
                labels = session.run("CALL db.labels()")
                for label in labels:
                    print(f"- {label[0]}")
                
                print("\n--- Relationship Types ---")
                rels = session.run("CALL db.relationshipTypes()")
                for rel in rels:
                    print(f"- {rel[0]}")
                
                print("\n--- Recent MENTIONED_IN Sample ---")
                sample = session.run("""
                    MATCH (n)-[r:MENTIONED_IN]->(d)
                    RETURN labels(n) as nodeLabels, type(r) as relType, labels(d) as docLabels, keys(d) as docKeys
                    LIMIT 5
                """)
                for record in sample:
                    print(f"Node labels: {record['nodeLabels']} -[:{record['relType']}]-> Doc labels: {record['docLabels']} (Keys: {record['docKeys']})")

                print("\n--- Lesson Node Sample ---")
                lessons = session.run("MATCH (l:Lesson) RETURN l.name as name")
                for record in lessons:
                    print(f"Lesson: {record['name']}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_labels_and_rels()
