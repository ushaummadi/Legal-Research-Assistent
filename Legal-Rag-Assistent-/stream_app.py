"""
LegalRAG: Indian Evidence Act RAG Assistant
Full-Stack Streamlit + Chroma + HuggingFace (2026)
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
os.environ.setdefault("DOCS_DIR", str(UPLOADS_DIR))      # point to uploads
os.environ.setdefault("UPLOADS_DIR", str(UPLOADS_DIR))

# ✅ App imports (after paths)
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
    # Prefer uploads folder, fallback to data/
    docs = load_documents(str(UPLOADS_DIR))
    if not docs:
        docs = load_documents(str(DATA_DIR))
    return docs


# ----------------------------
# MAIN APP
# ----------------------------
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
        config = yaml.load(f, Loader=SafeLoader) or {}

    config.setdefault("credentials", {}).setdefault("usernames", {})
    config.setdefault("cookie", {})

    # ----------------------------
    # ✅ AUTH
    # ----------------------------
    cookie_name = config["cookie"].get("name", "legalgpt_auth")
    cookie_expiry_days = float(config["cookie"].get("expiry_days", 30))
    cookie_key = (
        st.secrets.get("AUTH_COOKIE_KEY", "")
        if hasattr(st, "secrets")
        else ""
    ) or config["cookie"].get("key", "")

    if not cookie_key:
        st.error("❌ Missing cookie key. Add AUTH_COOKIE_KEY in Streamlit Cloud → Secrets.")
        st.stop()

    authenticator = stauth.Authenticate(
        config["credentials"],
        cookie_name,
        cookie_key,
        cookie_expiry_days=cookie_expiry_days,
    )

    try:
        name, authentication_status, username = authenticator.login(location="sidebar")
    except Exception:
        authentication_status = None

    if authentication_status is not True:
        st.sidebar.markdown("---")
        with st.sidebar.expander("👤 Account", expanded=True):
            tab_login, tab_signup = st.tabs(["Login", "Sign up"])

            with tab_login:
                name, authentication_status, username = authenticator.login(
                    location="sidebar",
                    fields={"Form name": "Login"}
                )

                if authentication_status is False:
                    st.error("❌ Wrong credentials")
                    st.stop()
                if authentication_status is None:
                    st.info("Please login to continue.")
                    st.stop()

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

    st.session_state["authentication_status"] = True
    st.session_state["username"] = username
    st.session_state["name"] = name

    # Theme CSS
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
        [data-testid="stSidebar"] { background-color: #171717 !important; }
        .stTextInput > div > div > div { background-color: #212121 !important; }
        .stButton > button { background-color: #212121 !important; color: #ececf1 !important; }
        * { border-color: #303030 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Session init
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []
    
    # Load history
    all_history = load_all_history()
    cur_sid = st.session_state["session_id"]
    if cur_sid not in all_history:
        all_history[cur_sid] = st.session_state["messages"]
        save_all_history(all_history)

    # ----------------------------
    # SIDEBAR
    # ----------------------------
    with st.sidebar:
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
        for sid in list(all_history.keys())[::-1]:
            msgs = all_history[sid]
            if not msgs: continue

            title = get_chat_title(msgs)
            is_selected = (sid == st.session_state["session_id"])
            c1, c2 = st.columns([1, 0.14], gap="small")
            with c1:
                t = "primary" if is_selected else "secondary"
                if st.button(title, key=f"load_{sid}", use_container_width=True, type=t):
                    st.session_state["session_id"] = sid
                    st.session_state["messages"] = msgs.copy()
                    st.rerun()
            with c2:
                if is_selected and st.button("✖", key=f"del_{sid}", type="secondary"):
                    del all_history[sid]
                    if sid == st.session_state["session_id"]:
                        st.session_state["session_id"] = str(uuid.uuid4())
                        st.session_state["messages"] = []
                        all_history[st.session_state["session_id"]] = []
                    save_all_history(all_history)
                    st.rerun()

        st.markdown("<div style='flex-grow: 1; height: 48vh;'></div>", unsafe_allow_html=True)
        
        # User Profile
        initials = (name[:2].upper() if name else "LG")
        st.markdown(
            f"""
            <div style='position: sticky; bottom: 0; width: 100%; background: #171717; border-top: 1px solid #303030; padding: 10px 12px;'>
              <div style='display: flex; align-items: center; gap: 10px;'>
                <div style='width: 36px; height: 36px; border-radius: 8px; background: #7b4ec9; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px;'>{initials}</div>
                <div><div style='color: #fff; font-size: 14px; font-weight: 600;'>{name}</div><div style='color: #b4b4b4; font-size: 12px;'>Free Plan</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        authenticator.logout(" Log out", "sidebar")

    # ----------------------------
    # MAIN PAGE
    # ----------------------------
    st.title("⚖️ LegalGPT")
    st.caption("Indian Evidence Act • Production RAG System")

    # ------------------------------------------------------------------
    # 🚀 ADMIN TOOLS / REBUILD INDEX (Moved to Main Page)
    # ------------------------------------------------------------------
    with st.expander("🛠️ **Admin Tools / Rebuild Index**", expanded=False):
        vsm = VectorStoreManager()
        
        col_stat, col_file = st.columns(2)
        with col_stat:
            st.metric("📊 Vectors Indexed", vsm.count())
        
        with col_file:
            st.write("**Found Files:**")
            files = list_source_files()
            if files:
                st.caption(", ".join(files[:3]) + "..." if len(files) > 3 else ", ".join(files))
            else:
                st.error("No files in data/uploads/")

        if st.button("🔄 **FORCE REBUILD INDEX NOW**", type="primary", use_container_width=True):
            with st.spinner("⏳ Indexing Evidence Act files..."):
                docs = load_docs_for_index()
                if not docs:
                    st.error("❌ No files found! Push Evidence txt files to GitHub data/uploads/ first.")
                else:
                    chunks = split_documents(docs)
                    vsm.add_documents(chunks)
                    st.success(f"✅ Success! {len(chunks)} chunks indexed. New vector count: {vsm.count()}")
                    st.rerun()
                    
        st.info("Click 'Force Rebuild' if you see 'No documents indexed'.")

    # ----------------------------
    # CHAT INTERFACE
    # ----------------------------
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask about Evidence Act, CrPC, IPC..."):
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("🔍 Analyzing legal documents..."):
                result = answer_question(query)
                answer = result.get("answer", "")
            placeholder.markdown(answer + "\n\n📚 *Powered by LegalRAG Pipeline*")

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        all_history = load_all_history()
        all_history[st.session_state["session_id"]] = st.session_state["messages"]
        save_all_history(all_history)
        st.rerun()

if __name__ == "__main__":
    run_streamlit_app()
