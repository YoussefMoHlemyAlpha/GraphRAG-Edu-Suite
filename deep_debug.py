import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
auth = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

def deep_debug():
    try:
        with GraphDatabase.driver(uri, auth=auth) as driver:
            with driver.session() as session:
                print("--- DOCUMENT NODES ---")
                docs = session.run("MATCH (d:Document) RETURN d")
                for record in docs:
                    node = record["d"]
                    print(f"Document Node Properties: {dict(node)}")

                print("\n--- MENTIONED_IN RELATIONSHIPS ---")
                rels = session.run("""
                    MATCH (n)-[r:MENTIONED_IN]->(d:Document)
                    RETURN labels(n) as l, n.id as id, type(r) as t, d.lesson as lesson
                    LIMIT 10
                """)
                count = 0
                for record in rels:
                    count += 1
                    print(f"Node {record['l']}({record['id']}) -[:{record['t']}]-> Document (lesson={record['lesson']})")
                print(f"Total MENTIONED_IN found (sample): {count}")

                print("\n--- NODES WITHOUT PART_OF ---")
                unlinked = session.run("""
                    MATCH (n) 
                    WHERE NOT n:Lesson AND NOT n:Document AND NOT (n)-[:PART_OF]->() 
                    RETURN count(n) as count
                """).single()["count"]
                print(f"Nodes missing PART_OF: {unlinked}")

                print("\n--- ALL LABELS ---")
                labels = session.run("CALL db.labels()")
                for label in labels:
                    print(f"- {label[0]}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    deep_debug()
