import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
auth = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

def find_any_link():
    try:
        with GraphDatabase.driver(uri, auth=auth) as driver:
            with driver.session() as session:
                print("--- SEARCHING FOR ANY LINK TO DOCUMENT ---")
                res = session.run("""
                    MATCH (n)-[r]->(d:Document)
                    RETURN labels(n) as l1, type(r) as t, labels(d) as l2, d.lesson as lesson
                    LIMIT 5
                """)
                found = False
                for record in res:
                    found = True
                    print(f"FOUND: {record['l1']} -[:{record['t']}]-> {record['l2']} (Lesson: {record['lesson']})")
                
                if not found:
                    print("No links to Document found at all.")
                    
                print("\n--- CHECKING DOCUMENT PROPERTIES ---")
                docs = session.run("MATCH (d:Document) RETURN d LIMIT 3")
                for record in docs:
                    print(f"Doc properties: {dict(record['d'])}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    find_any_link()
