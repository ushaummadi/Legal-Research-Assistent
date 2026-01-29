"""
LegalRAG: Indian Evidence Act RAG Assistant
Full-Stack Streamlit + Chroma + HuggingFace (2026) ✅ LOGIN PERFECT
"""

import os
import sys
import json
import uuid
from pathlib import Path

import streamlit as st
import yaml
from yaml.loader import SafeLoader

import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher

# ----------------------------
# PATHS (set BEFORE app imports)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
CONFIG_PATH = BASE_DIR / "config.yaml"
HISTORY_FILE = BASE_DIR / "chat_history.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ["OTEL_PYTHON_DISABLED"] = "true"
os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", str(CHROMA_DIR))
os.environ.setdefault("DOCS_DIR", str(UPLOADS_DIR))
os.environ.setdefault("UPLOADS_DIR", str(UPLOADS_DIR))

# ✅ App imports
from config.settings import settings
from src.ingestion.document_processor import load_documents, split_documents
from src.ingestion.vector_store import VectorStoreManager
from src.generation.rag_pipeline import answer_question

# ----------------------------
# HELPERS
# ----------------------------
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

def list_source_files():
    files = []
    if DATA_DIR.exists():
        files += [p.name for p in DATA_DIR.iterdir() if p.is_file()]
    if UPLOADS_DIR.exists():
        files += [p.name for p in UPLOADS_DIR.iterdir() if p.is_file()]
    return sorted(set(files))

def load_docs_for_index():
    docs = load_documents(str(UPLOADS_DIR))
    if not docs:
        docs = load_documents(str(DATA_DIR))
    return docs

# ----------------------------
# MAIN APP ✅ BULLETPROOF LOGIN + AUTO-LOGIN
# ----------------------------
def run_streamlit_app():
    st.set_page_config(
        page_title="LegalGPT - Evidence Act RAG",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ✅ CREATE config.yaml if missing
    if not CONFIG_PATH.exists():
        config = {"credentials": {"usernames": {}}, "cookie": {}}
        save_config(config)
        st.info("✅ Created config.yaml - Use signup to create first user!")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader) or {}
    config.setdefault("credentials", {}).setdefault("usernames", {})
    config.setdefault("cookie", {})

    # 🔥 AUTHENTICATION - COOKIE + SECRETS
    cookie_name = config["cookie"].get("name", "legalgpt_auth")
    cookie_expiry_days = float(config["cookie"].get("expiry_days", 30))
    cookie_key = (
        st.secrets.get("AUTH_COOKIE_KEY", "fallback_key_123")  # ✅ Fallback for local
        if hasattr(st, "secrets") 
        else "fallback_key_123"
    ) or config["cookie"].get("key", "fallback_key_123")

    authenticator = stauth.Authenticate(
        config["credentials"],
        cookie_name,
        cookie_key,
        cookie_expiry_days=cookie_expiry_days,
    )

    # ✅ BULLETPROOF SESSION INIT
    auth_keys = ["authentication_status", "name", "username"]
    for key in auth_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    # 🛑 LOGIN SCREEN (Clean logic)
    if st.session_state["authentication_status"] != "authenticated":
        st.sidebar.markdown("---")
        with st.sidebar.expander("👤 Account", expanded=True):
            tab_login, tab_signup = st.tabs(["Login", "Sign up"])

            with tab_login:
                name, authentication_status, username = authenticator.login(
                    location="main", 
                    fields={"Form name": "LegalGPT Login"}
                )

            with tab_signup:
                with st.form("signup_form", clear_on_submit=True):
                    new_fullname = st.text_input("Full Name")
                    new_user = st.text_input("Username")
                    new_pass = st.text_input("Password", type="password")
                    new_pass2 = st.text_input("Confirm Password", type="password")
                    
                    if st.form_submit_button("🚀 Create & Enter"):
                        if new_pass != new_pass2:
                            st.error("❌ Passwords don't match!")
                        elif new_user in config["credentials"]["usernames"]:
                            st.error("❌ Username exists!")
                        elif len(new_pass) < 6:
                            st.error("❌ Password too short (min 6 chars)!")
                        else:
                            # ✅ SAVE + AUTO-LOGIN
                            hashed = Hasher([new_pass]).generate()[0]
                            config["credentials"]["usernames"][new_user] = {
                                "name": new_fullname, "password": hashed
                            }
                            save_config(config)
                            
                            st.session_state["name"] = new_fullname
                            st.session_state["username"] = new_user
                            st.session_state["authentication_status"] = "authenticated"
                            st.success(f"✅ Welcome {new_fullname}! Loading LegalGPT...")
                            st.rerun()
        
        # STOP if not authenticated
        st.stop()
    
    # ✅ USER INFO (logged in)
    name = st.session_state["name"]
    username = st.session_state["username"]

    # Dark Theme CSS
    st.markdown("""
    <style>
    html, body, #root, .stApp, [data-testid="stSidebar"] { background-color: #171717 !important; }
    .stTextInput > div > div > div, .stTextArea > div > div > div { background-color: #212121 !important; }
    .stButton > button { background-color: #212121 !important; color: #ececf1 !important; border-radius: 6px; }
    .stMetric > div > div > div { color: #ececf1 !important; }
    * { border-color: #303030 !important; color: #ececf1 !important; }
    </style>
    """, unsafe_allow_html=True)

    # Chat Session Init
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []

    all_history = load_all_history()
    cur_sid = st.session_state["session_id"]
    if cur_sid not in all_history:
        all_history[cur_sid] = st.session_state["messages"]
        save_all_history(all_history)

    # 🔥 SIDEBAR - Chats + Profile + Logout
    with st.sidebar:
        if st.button("➕ New Chat", use_container_width=True, type="secondary"):
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.rerun()

        st.caption("💬 Your Chats")
        for sid in list(all_history.keys())[::-1]:
            msgs = all_history[sid]
            if not msgs: continue
            title = get_chat_title(msgs)
            is_selected = (sid == cur_sid)
            c1, c2 = st.columns([1, 0.15])
            with c1:
                if st.button(title, key=f"chat_{sid}", use_container_width=True, 
                           type="primary" if is_selected else "secondary"):
                    st.session_state["session_id"] = sid
                    st.session_state["messages"] = msgs.copy()
                    st.rerun()
            with c2:
                if is_selected and st.button("✖", key=f"del_{sid}", type="secondary"):
                    del all_history[sid]
                    st.session_state["session_id"] = str(uuid.uuid4())
                    st.session_state["messages"] = []
                    save_all_history(all_history)
                    st.rerun()
        
        st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
        
        # Profile + Logout
        initials = name[:2].upper() if name else "LG"
        st.markdown(f"""
        <div style='position: sticky; bottom: 20px; width: 100%; padding: 12px;'>
          <div style='display: flex; align-items: center; gap: 12px;'>
            <div style='width: 40px; height: 40px; border-radius: 8px; background: linear-gradient(135deg, #7b4ec9, #5a3fcc); 
                        color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 16px;'>
              {initials}
            </div>
            <div>
              <div style='color: #fff; font-size: 15px; font-weight: 600;'>{name}</div>
              <div style='color: #b4b4b4; font-size: 13px;'>@ {username} • Free Plan</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        authenticator.logout("🚪 Logout", "sidebar", key="main_logout")

    # MAIN CONTENT
    st.title("⚖️ LegalGPT")
    st.caption("🔍 Indian Evidence Act • Production RAG Assistant")

    # 🛠️ ADMIN PANEL
    with st.expander("🔧 **Admin: Index Management**", expanded=False):
        vsm = VectorStoreManager()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Vectors", vsm.count())
        with col2:
            st.metric("📁 Files", len(list_source_files()))
        with col3:
            if st.button("🔄 Rebuild", key="quick_rebuild"):
                st.rerun()
        
        st.caption("**Source Files:**")
        files = list_source_files()
        if files:
            st.code("\n".join(files[:10]), language="text")
        else:
            st.warning("📤 Add Evidence Act .txt files to `data/uploads/`")

        if st.button("🚀 **FORCE FULL REBUILD**", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing legal documents..."):
                docs = load_docs_for_index()
                if docs:
                    chunks = split_documents(docs)
                    vsm.add_documents(chunks)
                    st.success(f"✅ Indexed {len(chunks)} chunks! Total vectors: {vsm.count()}")
                else:
                    st.error("❌ No documents found! Add files first.")
                st.rerun()

    # 💬 CHAT INTERFACE
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # New message
    if prompt := st.chat_input("💭 Ask about Evidence Act (e.g. 'Section 32 dying declaration', 'murder evidence admissibility')..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching Evidence Act database..."):
                result = answer_question(prompt)
                response = result.get("answer", "No relevant sections found. Try 'Rebuild Index' or ask differently.")
            
            st.markdown(response + "\n\n📚 *LegalRAG • ChromaDB + HuggingFace*")

        st.session_state["messages"].append({"role": "assistant", "content": response})
        
        # Save history
        all_history[cur_sid] = st.session_state["messages"]
        save_all_history(all_history)
        st.rerun()

if __name__ == "__main__":
    run_streamlit_app()
