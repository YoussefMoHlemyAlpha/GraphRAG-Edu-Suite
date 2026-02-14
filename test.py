from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

print(f"DEBUG: URI being used is: {os.getenv('NEO4J_URI')}")

uri = os.getenv("NEO4J_URI")
auth = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

try:
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
    print("✅ Connection Successful!")
except Exception as e:
    print(f"❌ Connection Failed: {e}")