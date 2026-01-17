import json
import re
import google.generativeai as genai
import streamlit as st

class QuizEngine:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def _extract_json(self, text):
        text = text.strip()
        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            # Regex to fix trailing commas (common LLM error)
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            try:
                return json.loads(json_str)
            except:
                return None
        return None

    def generate_single_question(self, main_concept, difficulty, context, related_concepts=None):
        """
        Generates ONE question. 
        If Hard, it explicitly asks to combine 'main_concept' with 'related_concepts'.
        """
        
        if difficulty == "Hard" and related_concepts:
            # MULTI-CONCEPT PROMPT
            concept_list_str = ", ".join([main_concept] + related_concepts)
            task_desc = f"Create a complex 'Hard' difficulty question that REQUIRES understanding the relationship between these concepts: {concept_list_str}."
        else:
            # STANDARD PROMPT
            task_desc = f"Create a '{difficulty}' difficulty question specifically about: {main_concept}."

        prompt = f"""
        Context: {context}
        
        Task: {task_desc}
        
        Requirements:
        1. Output valid JSON only.
        2. The question must be unsolvable without understanding the core concept.
        
        JSON Format:
        {{
            "question": "...",
            "options": ["A)...", "B)...", "C)...", "D)..."],
            "correct_option": "A", 
            "difficulty": "{difficulty}",
            "explanation": "Detailed reasoning..."
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            data = self._extract_json(response.text)
            
            if data:
                data['concept'] = main_concept
                # Tag secondary concepts if hard
                if related_concepts:
                    data['related_concepts'] = related_concepts
                return data
            else:
                return None
        except Exception as e:
            print(f"Gen Error: {e}")
            return None

    def grade_answer(self, question, user_answer_char, context):
        prompt = f"""
        Question: {question['question']}
        Correct Answer: {question['correct_option']}
        Student Answer: {user_answer_char}
        Context: {context}
        
        Evaluate the answer. Output JSON:
        {{
            "score": <float 0.0 to 1.0>,
            "feedback": "<short explanation>",
            "misconceptions": ["<concept>"]
        }}
        """
        try:
            response = self.model.generate_content(prompt)
            data = self._extract_json(response.text)
            if data: return data
            raise ValueError("No JSON")
        except:
            # Fallback
            is_correct = user_answer_char.upper() == question['correct_option'][0].upper()
            return {"score": 1.0 if is_correct else 0.0, "feedback": "Grading failed, using answer key.", "misconceptions": []}

class StudentModel:
    def __init__(self, concepts, graph):
        # concepts: list of dicts {name, importance}
        # graph: networkx graph
        self.mastery = {c['name']: 0.5 for c in concepts}
        self.importance = {c['name']: c.get('importance', 0.5) for c in concepts}
        self.graph = graph
        self.alpha = 0.3 # Learning rate

    def update_mastery(self, concept, score):
        old_m = self.mastery.get(concept, 0.5)
        new_m = old_m + self.alpha * (score - old_m)
        self.mastery[concept] = max(0.0, min(1.0, new_m))
        
        if score < 0.5 and self.graph.has_node(concept):
            predecessors = list(self.graph.predecessors(concept))
            for pred in predecessors:
                if pred in self.mastery:
                    self.mastery[pred] = max(0.0, self.mastery[pred] - 0.05)

    def get_next_concept_strategy(self):
        best_concept = None
        max_priority = -1
        
        for name, m in self.mastery.items():
            imp = self.importance.get(name, 0.01)
            priority = (1 - m) * imp
            
            if priority > max_priority:
                max_priority = priority
                best_concept = name
                
        if not best_concept:
            # Fallback if list empty
            if self.mastery:
                return list(self.mastery.keys())[0], "Medium"
            return None, "Medium"

        current_m = self.mastery.get(best_concept, 0.5)
        if current_m < 0.4:
            difficulty = "Easy"
        elif current_m < 0.7:
            difficulty = "Medium"
        else:
            difficulty = "Hard"
            
        return best_concept, difficulty
