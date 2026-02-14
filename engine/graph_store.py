import os
import streamlit as st
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

class QuizGraphStore:
    def __init__(self):
        # Fallback order: environment variables (.env) -> Streamlit secrets
        uri = os.getenv("NEO4J_URI") or st.secrets.get("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME") or st.secrets.get("NEO4J_USERNAME")
        pwd = os.getenv("NEO4J_PASSWORD") or st.secrets.get("NEO4J_PASSWORD")

        if not all([uri, user, pwd]):
            st.error("Missing Neo4j configuration. Please check your .env or Streamlit Secrets.")
            raise ValueError("Incomplete Neo4j credentials.")

        try:
            self.graph = Neo4jGraph(
                url=uri,
                username=user,
                password=pwd
            )
            # A quick ping to confirm the connection is live
            self.graph.query("RETURN 1 AS connection_test")
            print("✅ QuizGraphStore: Connected to Aura successfully.")
        except Exception as e:
            print(f"❌ QuizGraphStore Error: {str(e)}")
            raise e

    # This is the missing piece that was causing your AttributeError!
    def query(self, query, params=None):
        """Standard wrapper to execute Cypher queries on the graph."""
        return self.graph.query(query, params)
    
    def get_lessons(self):
        """Returns a list of lesson names already in the graph."""
        query = "MATCH (l:Lesson) RETURN l.name AS name"
        results = self.query(query)
        return [r['name'] for r in results]

    def wipe_database(self):
        """Deletes all nodes and relationships from the database."""
        query = "MATCH (n) DETACH DELETE n"
        self.query(query)