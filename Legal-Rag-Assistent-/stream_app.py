"""
LegalRAG: Indian Evidence Act RAG Assistant
Full-Stack Streamlit + Chroma + HuggingFace (2026) - PRODUCTION READY
"""
import sys
import json
import uuid
from pathlib import Path

# ✅ CRITICAL: Fix SQLite for Streamlit Cloud
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.hasher import Hasher

# App imports
from src.ingestion.document_processor import load_documents, split_documents
from src.ingestion.vector_store import VectorStoreManager
from src.generation.rag_pipeline import answer_question

# --------------------------------------------------------------------Here's your **COMPLETE FIXED `stream_app.py`** with all bugs resolved:

```python
"""
LegalRAG: Indian Evidence Act RAG Assistant
Full-Stack Streamlit + Chroma + HuggingFace (2026) - PRODUCTION READY
"""
import sys
import json
import uuid
from pathlib import Path

# ✅ CRITICAL: Fix SQLite for Streamlit Cloud + Chroma
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # Local dev doesn't need this

import streamlit as st
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.hasher import Hasher

# App imports
from src.ingestion.document_processor import load_documents, split_documents
from src.ingestion.vector_store import VectorStoreManager
from src.generation.rag_pipeline import answer_question

# --------------------------------------------------------------------
# PATHS (Absolute for Cloud)
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
CONFIG_PATH = BASE_DIR / "config.yaml"
HISTORY_FILE = BASE_DIR / "chat_history.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# --------------------------------------------------------------------
# Helpers
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
    CONFIG_PATH.write_text(
        yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

def ensure_dirs():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------
def run_streamlit_app():
    st.set_page_config(
        page_title="LegalGPT - Evidence Act RAG",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    ensure_dirs()

    if not CONFIG_PATH.exists():
        st.error("❌ config.yaml not found!")
        st.stop()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader) or {}

    # CSS (unchanged)
    st.markdown(
        """
        <style>
        html, body, #root, .stApp,
        header[data-testid="stHeader"],
        footer[data-testid="stFooter"],
        section[data-testid="stAppViewContainer"],
        section[data-testid="stChatInputContainer"],
        .stApp > div > div > div[class*="main"],
        .block-container,
        [data-testid="stSidebar"] {
            background-color: #171717 !important;
        }
        
        [data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] { background-color: #212121 !important; }
        [data-testid="stSidebar"] .stTabs [data-baseweb="tab"] { background-color: transparent !important; color: #ececf1 !important; }
        
        .stChatInput > div > div { background-color: transparent !important; }
        .stChatMessage, [data-testid="stChatMessage"] { background-color: transparent !important; }
        
        [data-testid="metric-container"], [data-testid="stHorizontalBlock"],
        section[data-testid="stSidebar"] div.element-container { background-color: #171717 !important; }
        
        .stTextInput > div > div > div { background-color: #212121 !important; }
        .stButton > button { background-color: #212121 !important; color: #ececf1 !important; }
        * { border-color: #303030 !important; }
        
        section[data-testid="stSidebar"] .block-container{ padding-top: 0.6rem; }
        [data-testid="stSidebar"] div.stButton{ margin-bottom: 0.12rem !important; }
        [data-testid="stSidebar"] [data-testid="column"]{ padding-left: 0.05rem !important; padding-right: 0.05rem !important; }
        
        [data-testid="stSidebar"] button[kind="tertiary"]{
          background: transparent !important;
          border: none !important;
          color: #ececf1 !important;
          border-radius: 10px !important;
        }
        [data-testid="stSidebar"] button[kind="tertiary"]:hover{
          background: #2a2a2a !important;
        }
        
        [data-testid="stSidebar"] button[kind="secondary"]{
          background: #353545 !important;
          border: none !important;
          color: #ffffff !important;
          border-radius: 10px !important;
        }
        
        [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:first-child button[kind="secondary"]{
          border-top-right-radius: 0px !important;
          border-bottom-right-radius: 0px !important;
        }
        [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:last-child button[kind="secondary"]{
          border-top-left-radius: 0px !important;
          border-bottom-left-radius: 0px !important;
          width: 38px !important;
          min-width: 38px !important;
          padding: 0px !important;
          font-weight: 900 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # AUTH
    config.setdefault("credentials", {}).setdefault("usernames", {})
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

    name = st.session_state["name"]

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

    # SIDEBAR
    with st.sidebar:
        st.markdown("### 🔎 Debug RAG Status")
        upload_count = len(list(UPLOADS_DIR.glob("*")))
        chroma_files_count = len(list(CHROMA_DIR.glob("*")))
        st.metric("📂 Upload files", upload_count)
        st.metric("🗄️ Chroma files", chroma_files_count)

        if st.button("🧪 Test Vector Count", key="test_vectors", use_container_width=True, type="secondary"):
            try:
                vsm = VectorStoreManager(persist_dir=str(CHROMA_DIR))
                count = vsm.count()
                if count > 0:
                    st.success(f"✅ {count:,} vectors ready!")
                else:
                    st.warning("⚠️ 0 vectors - Rebuild Index!")
            except Exception as e:
                st.error(f"❌ {str(e)[:100]}")

        if st.button("➕ New chat", use_container_width=True, type="secondary"):
            current_sid = st.session_state.get("session_id")
            current_msgs = st.session_state.get("messages", [])
            if current_sid and current_msgs:
                all_history[current_sid] = current_msgs
                save_all_history(all_history)

            new_sid = str(uuid.uuid4())
            st.session_state["session_id"] = new_sid
            st.session_state["messages"] = []
            all_history.setdefault(new_sid, [])
            save_all_history(all_history)
            st.rerun()

        st.caption("Your chats")

        # ✅ FIXED: Crash-proof columns
        for sid in list(all_history.keys())[::-1]:
            msgs = all_history[sid]
            if not msgs:
                continue

            title = get_chat_title(msgs)
            is_selected = (sid == st.session_state["session_id"])

            # FIXED: No gap/vertical_alignment parameters
            c1, c2 = st.columns([0.85, 0.15])

            with c1:
                t = "secondary" if is_selected else "tertiary"
                if st.button(title, key=f"load_{sid}", use_container_width=True, type=t):
                    current_sid = st.session_state.get("session_id")
                    current_msgs = st.session_state.get("messages", [])
                    if current_sid is not None:
                        all_history[current_sid] = current_msgs
                        save_all_history(all_history)

                    st.session_state["session_id"] = sid
                    st.session_state["messages"] = msgs.copy()
                    st.rerun()

            with c2:
                if is_selected:
                    if st.button("✖", key=f"del_{sid}", type="secondary"):
                        if sid in all_history:
                            del all_history[sid]
                        if sid == st.session_state["session_id"]:
                            new_sid = str(uuid.uuid4())
                            st.session_state["session_id"] = new_sid
                            st.session_state["messages"] = []
                            all_history[new_sid] = []
                        save_all_history(all_history)
                        st.rerun()
                else:
                    st.write("")

        st.markdown("<div style='flex-grow: 1; height: 48vh;'></div>", unsafe_allow_html=True)

        # Profile footer
        initials = (name[:2].upper() if name else "LG")
        st.markdown(
            f"""
        <div style='position: sticky; bottom: 0; width: 100%; background: #171717; border-top: 1px solid #303030; padding: 10px 12px;'>
          <div style='display: flex; align-items: center; gap: 10px;'>
            <div style='width: 36px; height: 36px; border-radius: 8px; background: #7b4ec9; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px;'>{initials}</div>
            <div>
              <div style='color: #fff; font-size: 14px; font-weight: 600;'>{name}</div>
              <div style='color: #b4b4b4; font-size: 12px;'>Free Plan</div>
            </div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("🚪 Log out", use_container_width=True):
            st.session_state["authentication_status"] = None
            st.session_state["username"] = None
            st.session_state["name"] = None
            st.rerun()

    # MAIN CONTENT
    st.title("⚖️ LegalGPT")
    st.caption("Indian Evidence Act -  Production RAG System")

    if show_settings:
        st.markdown("---")
        st.subheader("⚙️ Settings")

        col_close, _ = st.columns([0.1, 1])
        with col_close:
            if st.button("✖", key="close_settings"):
                st.query_params.clear()
                st.rerun()

        # ✅ FIXED: Proper indexing with debug
        if st.button("🔄 Rebuild Index", use_container_width=True, type="primary"):
            with st.spinner("⏳ Indexing..."):
                try:
                    # Load docs
                    docs = load_documents(str(UPLOADS_DIR))
                    st.info(f"✅ Loaded {len(docs)} documents")
                    
                    if not docs:
                        st.error(f"❌ No files in {UPLOADS_DIR}")
                    else:
                        # Split
                        chunks = split_documents(docs)
                        st.info(f"✅ Created {len(chunks)} chunks")
                        
                        # Index
                        vsm = VectorStoreManager(persist_dir=str(CHROMA_DIR))
                        vsm.add_documents(chunks)
                        
                        final_count = vsm.count()
                        st.success(f"✅ **COMPLETE!** {final_count:,} vectors indexed")
                        st.balloons()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        if st.button("🗑️ Clear History", use_container_width=True):
            save_all_history({})
            st.session_state["messages"] = []
            st.session_state["session_id"] = str(uuid.uuid4())
            all_history = {st.session_state["session_id"]: []}
            save_all_history(all_history)
            st.rerun()

        st.markdown("---")

    # Chat UI
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask about Evidence Act, CrPC, IPC..."):
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("🔍 Searching..."):
                try:
                    result = answer_question(query, chroma_dir=str(CHROMA_DIR))
                    answer = result.get("answer", "No answer generated")
                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
            placeholder.markdown(answer + "\n\n📚 *LegalRAG Pipeline*")

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        all_history = load_all_history()
        all_history[st.session_state["session_id"]] = st.session_state["messages"]
        save_all_history(all_history)
        st.rerun()


if __name__ == "__main__":
    run_streamlit_app()
