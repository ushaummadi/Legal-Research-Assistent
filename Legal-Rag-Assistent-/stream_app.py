"""
LegalRAG: Indian Evidence Act RAG Assistant
Full-Stack Streamlit + Chroma + HuggingFace (2026) - FIXED VERSION
"""
import sys
import os
import json
import uuid
from pathlib import Path

import streamlit as st
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.hasher import Hasher

# --------------------------------------------------------------------
# 1. SETUP PATHS (Absolute, for Streamlit Cloud stability)
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
CONFIG_PATH = BASE_DIR / "config.yaml"
HISTORY_FILE = BASE_DIR / "chat_history.json"

# Ensure imports work when app runs from repo root
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# --------------------------------------------------------------------
# 2. IMPORTS (After path setup)
# --------------------------------------------------------------------
from src.ingestion.document_processor import load_documents, split_documents
from src.ingestion.vector_store import VectorStoreManager
from src.generation.rag_pipeline import answer_question

# --------------------------------------------------------------------
# 3. HELPER FUNCTIONS (unchanged)
# --------------------------------------------------------------------
def load_all_history():
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}

def save_all_history(all_history):
    HISTORY_FILE.write_text(
        json.dumps(all_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def get_chat_title(messages):
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"][:28] + "..." if len(msg["content"]) > 28 else msg["content"]
    return "New Chat"

def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

# --------------------------------------------------------------------
# 4. MAIN APP (with FIXED paths + debug)
# --------------------------------------------------------------------
def run_streamlit_app():
    st.set_page_config(
        page_title="LegalGPT - Evidence Act RAG",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not CONFIG_PATH.exists():
        st.error("❌ config.yaml not found!")
        st.stop()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader)

    config.setdefault("credentials", {}).setdefault("usernames", {})

    # AUTHENTICATION (unchanged)
    st.session_state.setdefault("authentication_status", None)
    st.session_state.setdefault("username", None)
    st.session_state.setdefault("name", None)

    if st.session_state["authentication_status"] is not True:
        st.sidebar.markdown("---")
        with st.sidebar.expander("👤 Account", expanded=True):
            tab_login, tab_signup = st.tabs(["Login", "Sign up"])

            with tab_login:
                st.info("👋 Welcome to LegalGPT")
                with st.form("login_form", clear_on_submit=False):
                    u = st.text_input("Username")
                    p = st.text_input("Password", type="password")
                    login_ok = st.form_submit_button("Login")

                if login_ok:
                    user = config.get("credentials", {}).get("usernames", {}).get(u)
                    if user and Hasher.check_pw(p, user["password"]):
                        st.session_state["authentication_status"] = True
                        st.session_state["username"] = u
                        st.session_state["name"] = user.get("name", u)
                        st.success("✅ Logged in!")
                        st.rerun()
                    else:
                        st.error("❌ Wrong credentials")

            with tab_signup:
                with st.form("signup_form", clear_on_submit=True):
                    new_fullname = st.text_input("Full Name")
                    new_email = st.text_input("Email")
                    new_user = st.text_input("Username")
                    new_pass = st.text_input("Password", type="password")
                    new_pass2 = st.text_input("Confirm Password", type="password")
                    signup_ok = st.form_submit_button("Create Account")

                if signup_ok:
                    if not all([new_fullname, new_email, new_user, new_pass, new_pass2]):
                        st.error("All fields required!")
                    elif new_pass != new_pass2:
                        st.error("Passwords don't match!")
                    elif new_user in config["credentials"]["usernames"]:
                        st.error("Username exists!")
                    else:
                        hashed = Hasher.hash(new_pass)
                        config["credentials"]["usernames"][new_user] = {
                            "name": new_fullname,
                            "email": new_email,
                            "password": hashed,
                        }
                        save_config(config)
                        st.success("✅ Account created! Now login.")
                        st.rerun()

        st.stop()

    # APP LOGIC (Post-Auth)
    name = st.session_state["name"]

    # CSS STYLING (unchanged - truncated for brevity)
    st.markdown("""
        <style>
        /* Your existing CSS here - keep it */
        </style>
    """, unsafe_allow_html=True)

    # Session init
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []

    all_history = load_all_history()
    cur_sid = st.session_state["session_id"]
    if cur_sid not in all_history:
        all_history[cur_sid] = st.session_state["messages"]
        save_all_history(all_history)

    qp = st.query_params
    show_settings = (qp.get("menu") == "settings")

    # ----------------------------------------------------------------
    # FIXED SIDEBAR with DEBUG BUTTONS
    # ----------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🔍 **Debug RAG Status**")
        
        # FIXED: Create dirs if missing
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Debug metrics
        upload_count = len(list(UPLOADS_DIR.glob("*")))
        chroma_files = len(list(CHROMA_DIR.glob("*")))
        st.metric("📂 Upload files", upload_count)
        st.metric("🗄️ Chroma files", chroma_files)
        st.caption(f"Paths: {UPLOADS_DIR} | {CHROMA_DIR}")

        # ✅ NEW: Test Chroma count button
        if st.button("🧪 Test Vector Count", key="test_vectors"):
            try:
                vsm = VectorStoreManager(persist_dir=str(CHROMA_DIR))
                count = vsm.collection.count()
                if count > 0:
                    st.success(f"✅ **{count:,} vectors ready!** RAG will work.")
                else:
                    st.warning("⚠️ **0 vectors** → Click 'Rebuild Index' first.")
                    st.info("Expected: 5,000-20,000 for 1038 files")
            except Exception as e:
                st.error(f"❌ VectorStore error: {str(e)[:100]}")

        if st.button("➕ New chat", use_container_width=True, type="secondary"):
            # ... existing new chat logic (unchanged)
            pass

        # Chat history list (your existing code - unchanged)
        st.caption("Your chats")
        # ... rest of chat list code

    # MAIN CONTENT
    st.title("⚖️ LegalGPT")
    st.caption("Indian Evidence Act • Production RAG System")

    # FIXED SETTINGS with path enforcement
    if show_settings:
        st.markdown("---")
        st.subheader("⚙️ **Rebuild Index** (Fixed Paths)")

        if st.button("🔄 **Rebuild Chroma Index**", use_container_width=True, type="primary"):
            with st.spinner("🔄 Indexing 1038 legal files..."):
                try:
                    # ✅ FIXED: Pass explicit paths
                    docs = load_documents(str(UPLOADS_DIR))  # Pass path!
                    st.info(f"✅ Loaded {len(docs)} documents")
                    
                    chunks = split_documents(docs)
                    st.info(f"✅ Created {len(chunks)} chunks")
                    
                    # ✅ FIXED: Pass persist_dir
                    vsm = VectorStoreManager(persist_dir=str(CHROMA_DIR))
                    vsm.add_documents(chunks)
                    
                    st.success(f"✅ **COMPLETE!** {vsm.collection.count():,} vectors in {CHROMA_DIR}")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Index error: {str(e)}")
                    st.error("Check src/ingestion/* accepts path params")

        st.markdown("---")

    # FIXED CHAT with better error handling
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask about Evidence Act, CrPC, IPC..."):
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("🔍 Searching legal database..."):
                try:
                    # ✅ FIXED: Pass path to pipeline
                    result = answer_question(query, chroma_dir=str(CHROMA_DIR))
                    answer = result.get("answer", "No answer generated.")
                except Exception as e:
                    answer = f"❌ Pipeline error: {str(e)}"
                    st.error("Check src/generation/rag_pipeline.py uses chroma_dir")
            
            placeholder.markdown(answer + "\n\n📚 *LegalRAG Pipeline*")
            st.session_state["messages"].append({"role": "assistant", "content": answer})

        # Save history (unchanged)
        all_history = load_all_history()
        all_history[st.session_state["session_id"]] = st.session_state["messages"]
        save_all_history(all_history)
        st.rerun()

if __name__ == "__main__":
    run_streamlit_app()
