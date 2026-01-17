import os
import time
import uuid
import networkx as nx
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier

# Fix OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from rag_engine import RAGEngine
from concept_engine import ConceptEngine
from quiz_engine import QuizEngine, StudentModel

st.set_page_config(page_title="Adaptive AI Quiz Generator", layout="wide")

# --- Constants ---
SCORE_WEIGHTS = {"Easy": 10, "Medium": 20, "Hard": 50}

# --- Session Init (Persistent Variables) ---
if "api_key" not in st.session_state: st.session_state.api_key = ""
if "rag" not in st.session_state: st.session_state.rag = None
if "student" not in st.session_state: st.session_state.student = None
if "long_term_memory" not in st.session_state: st.session_state.long_term_memory = [] 
if "stage" not in st.session_state: st.session_state.stage = "setup"

# --- Session Init (Transient Variables) ---
if "history" not in st.session_state: st.session_state.history = []
if "total_score" not in st.session_state: st.session_state.total_score = 0
if "count" not in st.session_state: st.session_state.count = 0

# --- HELPER: Reset Logic ---
def reset_quiz_state():
    """
    Clears current quiz but saves data for ML calibration.
    """
    # Archive current history to Long Term Memory
    if "history" in st.session_state and st.session_state.history:
        for item in st.session_state.history:
            st.session_state.long_term_memory.append({
                "time": item.get('time', 0),
                "diff": item.get('diff', "Medium"),
                "correct": item.get('correct', False)
            })

    # Reset Quiz Variables
    st.session_state.history = []
    st.session_state.total_score = 0
    st.session_state.count = 0
    st.session_state.current_q = None
    st.session_state.last_result = None
    st.session_state.last_time = 0
    st.session_state.stage = "config"


# --- CORE LOGIC: Hybrid Adaptive Strategy ---
def get_next_difficulty_hybrid(current_history, long_term_memory):
    """
    HYBRID LOGIC:
    1. Questions 0-10: Strict Heuristic Rules (Calibration Phase)
    2. Questions 10+: ML Decision Tree (Personalized Phase)
    """
    
    # 1. Combine Data to check Total Experience
    all_data = long_term_memory + current_history
    total_questions_seen = len(all_data)
    
    # Get the LAST interaction to make immediate decisions
    if current_history:
        last_q = current_history[-1]
    elif long_term_memory:
        last_q = long_term_memory[-1]
    else:
        return "Medium" # Absolute Start

    # ============================================================
    # PHASE 1: STRICT PARAMETERS (Calibration)
    # Strict rules to gather clean data for the first 10 questions.
    # ============================================================
    if total_questions_seen < 10:
        
        # Rule 1: Accuracy is King. If you fail, you drop.
        if not last_q['correct']:
            return "Easy"
            
        # Rule 2: Speed Gates (Flow State)
        # If Correct:
        if last_q['time'] < 15:
            return "Hard"      # Fast = Master
        elif last_q['time'] > 45:
            return "Medium"    # Slow = Struggling, don't bump up
        else:
            return "Hard"      # Average speed = Bias up for challenge

    # ============================================================
    # PHASE 2: ML MODEL (Personalized)
    # The Decision Tree takes over once we have >10 data points.
    # ============================================================
    
    # Prepare Training Data
    diff_map = {"Easy": 1, "Medium": 2, "Hard": 3}
    X = []
    y = []
    
    for i in range(1, len(all_data)):
        prev = all_data[i-1]
        curr = all_data[i]
        
        # Guard against bad data
        if 'time' not in prev or 'diff' not in prev: continue

        features = [
            prev['time'],            
            diff_map.get(prev['diff'], 2),  
            int(prev['correct'])     
        ]
        X.append(features)
        y.append(int(curr['correct']))

    # Safety: If ML data extraction failed, fallback to rules
    if len(X) < 5: return "Medium"
    
    try:
        # Train Tree
        clf = DecisionTreeClassifier(max_depth=4) # Slightly deeper tree for more data
        clf.fit(X, y)

        # Predict based on current state
        current_context = [
            last_q['time'], 
            diff_map.get(last_q['diff'], 2), 
            int(last_q['correct'])
        ]
        
        # Predict success probability (1=Pass, 0=Fail)
        pred_success = clf.predict([current_context])[0]
        
        # ML Policy:
        # If model thinks you will Pass -> Increase/Maintain Challenge
        # If model thinks you will Fail -> Decrease Challenge
        return "Hard" if pred_success == 1 else "Medium"
        
    except Exception as e:
        print(f"ML Error: {e}")
        return "Medium"

# --- HELPER: Multi-Concept Finder ---
def get_overlapping_concepts(graph, main_concept, count=2):
    if not graph.has_node(main_concept): return []
    neighbors = list(graph.successors(main_concept)) + list(graph.predecessors(main_concept))
    if len(neighbors) < count:
        all_nodes = list(graph.nodes())
        neighbors += [n for n in all_nodes if n != main_concept][:count]
    return list(set(neighbors))[:count]

# --- SIDEBAR: Setup & Upload ---
with st.sidebar:
    st.title("⚙️ Adaptive Engine")
    
    if st.session_state.rag is None:
        api_key = st.text_input("API Key", type="password")
        if api_key: st.session_state.api_key = api_key
        f = st.file_uploader("Upload PDF", type="pdf")
        
        if st.button("Initialize System") and f and api_key:
            with st.spinner("Ingesting Book..."):
                rag = RAGEngine(api_key)
                txt, _ = rag.process_pdf(f)
                st.session_state.rag = rag
                
                ce = ConceptEngine(api_key)
                g_data = ce.extract_concept_graph(txt)
                st.session_state.graph_data = g_data
                st.session_state.nx_graph = ce.build_network_graph(g_data)
                
                st.session_state.student = StudentModel(g_data['concepts'], st.session_state.nx_graph)
                st.session_state.quiz_engine = QuizEngine(api_key)
                st.session_state.stage = "config"
                st.rerun()
    else:
        st.success("📚 Book Active")
        # Show "Phase" Indicator
        total_qs = len(st.session_state.long_term_memory) + len(st.session_state.history)
        if total_qs < 10:
            st.info(f"Phase: Calibration ({total_qs}/10)")
        else:
            st.success(f"Phase: AI Optimized ({total_qs} data points)")
            
        if st.button("Reset Book & System"):
            st.session_state.clear()
            st.rerun()

# --- MAIN LOGIC ---

if st.session_state.stage == "config":
    st.title("📝 Quiz Configuration")
    if st.session_state.student is None:
        st.warning("Please upload a book first.")
    else:
        concepts = [c['name'] for c in st.session_state.graph_data['concepts']]
        sel = st.multiselect("Select Topics:", concepts, default=concepts[:min(3, len(concepts))])
        num = st.number_input("Questions:", 3, 50, 5)
        
        if st.button("Start Quiz"):
            st.session_state.config = {"concepts": sel, "total": num}
            st.session_state.count = 0
            st.session_state.total_score = 0
            st.session_state.history = []
            st.session_state.stage = "active"
            st.rerun()

elif st.session_state.stage == "active":
    conf = st.session_state.config
    if st.session_state.count >= conf['total']:
        st.session_state.stage = "report"
        st.rerun()

    # --- Generate Question ---
    if st.session_state.get("current_q") is None:
        
        # A. HYBRID STRATEGY CALL
        target_diff = get_next_difficulty_hybrid(st.session_state.history, st.session_state.long_term_memory)
        
        # B. Get Concept
        best_c, _ = st.session_state.student.get_next_concept_strategy() 
        target_concept = best_c if best_c in conf['concepts'] else conf['concepts'][0]
        import random
        if st.session_state.student.mastery.get(target_concept, 0) > 0.8:
            target_concept = random.choice(conf['concepts'])

        # C. Hard Logic
        related = []
        if target_diff == "Hard":
            related = get_overlapping_concepts(st.session_state.nx_graph, target_concept, 2)
            if related:
                st.toast(f"🔥 Hard Mode: {target_concept} + {related}")

        # D. Generate
        with st.spinner(f"AI ({target_diff}): {target_concept}..."):
            context = st.session_state.rag.get_context(target_concept)
            if related:
                for r in related: context += "\n" + st.session_state.rag.get_context(r)
            
            q_data = st.session_state.quiz_engine.generate_single_question(
                target_concept, target_diff, context, related
            )
            
            if q_data:
                q_data['id'] = str(uuid.uuid4())
                st.session_state.current_q = q_data
                st.session_state.q_start_time = time.time()
                st.rerun()
            else:
                st.error("Generation failed. Retrying...")
                time.sleep(1)
                st.rerun()

    # --- Render Question ---
    q = st.session_state.current_q
    col1, col2 = st.columns([3, 1])
    col1.subheader(f"Q{st.session_state.count + 1}: {q['question']}")
    col2.metric("Difficulty", q['difficulty'], f"{SCORE_WEIGHTS.get(q['difficulty'], 10)} pts")
    
    with st.form("ans_form"):
        choice = st.radio("Options:", q['options'])
        sub = st.form_submit_button("Submit")
        
        if sub:
            time_taken = time.time() - st.session_state.q_start_time
            user_char = choice.split(")")[0] if ")" in choice else choice[0]
            
            ctx = st.session_state.rag.get_context(q['concept'])
            res = st.session_state.quiz_engine.grade_answer(q, user_char, ctx)
            
            st.session_state.count += 1
            st.session_state.total_score += (SCORE_WEIGHTS.get(q['difficulty'], 10) * res['score'])
            st.session_state.student.update_mastery(q['concept'], res['score'])
            
            st.session_state.history.append({
                "q": q['question'],
                "correct": res['score'] == 1.0,
                "time": round(time_taken, 1),
                "diff": q['difficulty'],
                "concept": q['concept']
            })
            
            if res['score'] == 1.0: st.success(f"Correct! ({time_taken:.1f}s)")
            else: st.error(f"Incorrect. {res['feedback']}")
            
            st.session_state.current_q = None
            time.sleep(2)
            st.rerun()

elif st.session_state.stage == "report":
    st.title("🏆 Quiz Results")
    st.metric("Final Score", st.session_state.total_score)
    
    df = pd.DataFrame(st.session_state.history)
    if not df.empty:
        st.dataframe(df[['q', 'diff', 'correct', 'time']])
        fig = px.scatter(df, x="time", y="diff", color="correct", category_orders={"diff": ["Easy", "Medium", "Hard"]})
        st.plotly_chart(fig)
    
    st.divider()
    st.subheader("🧠 Long-Term Concept Mastery")
    active_concepts = {k:v for k,v in st.session_state.student.mastery.items() if v != 0.5}
    if active_concepts:
        st.bar_chart(pd.DataFrame(list(active_concepts.items()), columns=["Concept", "Mastery"]).set_index("Concept"))

    if st.button("Start New Quiz (Keep Learning Data)"):
        reset_quiz_state()
        st.rerun()
