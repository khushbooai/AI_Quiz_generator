import json
import networkx as nx
import google.generativeai as genai
from collections import Counter

class ConceptEngine:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def extract_concept_graph(self, full_text):
        """
        Extracts concepts and relationships from text using Gemini.
        Returns a JSON object with concepts (w/ frequency) and edges.
        """
        # Truncate text if too large for prompt context window (simple safety measure)
        # In production, you might map-reduce over chunks.
        truncated_text = full_text[:30000] 
        
        prompt = f"""
        Analyze the following educational text. Identify key concepts and their relationships.
        Output ONLY valid JSON in the following format, no markdown formatting:
        {{
            "concepts": [
                {{"name": "Concept Name", "frequency": <estimated_occurrence_count_int>}}
            ],
            "edges": [
                {{"from": "Concept A", "to": "Concept B", "relation": "implies/requires/part_of"}}
            ]
        }}
        
        Text to analyze:
        {truncated_text}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Cleanup JSON text if Gemini adds markdown backticks
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            
            # Normalize Importance
            total_freq = sum(c['frequency'] for c in data['concepts'])
            for c in data['concepts']:
                c['importance'] = c['frequency'] / total_freq if total_freq > 0 else 0
                
            return data
        except Exception as e:
            print(f"Error extracting concepts: {e}")
            return {"concepts": [], "edges": []}

    def build_network_graph(self, graph_data):
        """Creates a NetworkX graph for propagation logic."""
        G = nx.DiGraph()
        for c in graph_data['concepts']:
            G.add_node(c['name'], importance=c['importance'])
        for e in graph_data['edges']:
            G.add_edge(e['from'], e['to'], relation=e['relation'])
        return G
