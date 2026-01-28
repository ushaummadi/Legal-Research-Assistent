
"""
LegalRAG: Indian Evidence Act RAG Assistant
Full-Stack Streamlit + Chroma + HuggingFace (2026)
"""
import sys
from pathlib import Path

import streamlit as st
import json
import uuid
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.hasher import Hasher

# ✅ FIXED IMPORTS (LangChain v1+ 2026)
from config.settings import settings
from src.ingestion.document_processor import load_documents, split_documents
from src.ingestion.vector_store import VectorStoreManager
from src.generation.rag_pipeline import answer_question

# --- PATHS ---
CONFIG_PATH = Path("config.yaml")
HISTORY_FILE = Path("chat_history.json")


# --- HELPER FUNCTIONS ---
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


# --- MAIN APP ---
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

    # ✅ SINGLE BOX AUTH: Login + Signup tabs
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
    username = st.session_state["username"]
    authentication_status = st.session_state["authentication_status"]

    # ✅ TOTAL #171717 EVERYWHERE
    st.markdown(
        """
        <style>
        /* TOTAL UNIFORM #171717 */
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
        
        /* Auth tabs styling */
        [data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] {
            background-color: #212121 !important;
        }
        [data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            color: #ececf1 !important;
        }
        
        /* Chat input */
        .stChatInput > div > div {
            background-color: transparent !important;
        }
        
        /* Chat areas */
        .stChatMessage, [data-testid="stChatMessage"] {
            background-color: transparent !important;
        }
        
        /* All elements match */
        [data-testid="metric-container"], 
        [data-testid="stHorizontalBlock"],
        section[data-testid="stSidebar"] div.element-container {
            background-color: #171717 !important;
        }
        /* Inputs, buttons, expanders */
        .stTextInput > div > div > div {
            background-color: #212121 !important;
        }
        .stButton > button {
            background-color: #212121 !important;
            color: #ececf1 !important;
        }
        /* Remove all borders */
        * {
            border-color: #303030 !important;
        }

        /* Compact sidebar padding */
        section[data-testid="stSidebar"] .block-container{ padding-top: 0.6rem; }

        /* Reduce vertical gap between history rows */
        [data-testid="stSidebar"] div.stButton{ margin-bottom: 0.12rem !important; }

        /* Reduce column padding inside sidebar rows */
        [data-testid="stSidebar"] [data-testid="column"]{ padding-left: 0.05rem !important; padding-right: 0.05rem !important; }

        /* Title buttons base (unselected = transparent via type="tertiary") */
        [data-testid="stSidebar"] button[kind="tertiary"]{
          background: transparent !important;
          border: none !important;
          box-shadow: none !important;
          color: #ececf1 !important;
          border-radius: 10px !important;
        }
        [data-testid="stSidebar"] button[kind="tertiary"]:hover{
          background: #2a2a2a !important;
          color: #fff !important;
        }

        /* Selected (type="secondary") -> light box */
        [data-testid="stSidebar"] button[kind="secondary"]{
          background: #353545 !important;
          border: none !important;
          box-shadow: none !important;
          color: #ffffff !important;
          border-radius: 10px !important;
        }

        /* Make (title + X) look like one combined box when selected */
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

    qp = st.query_params
    show_settings = (qp.get("menu") == "settings")

    # SIDEBAR (post-auth)
    with st.sidebar:
        if st.button("➕ New chat", use_container_width=True, type="secondary"):
            current_sid = st.session_state.get("session_id")
            current_msgs = st.session_state.get("messages", [])
            if current_sid and current_msgs:
                all_history[current_sid] = current_msgs

            new_sid = str(uuid.uuid4())
            st.session_state["session_id"] = new_sid
            st.session_state["messages"] = []
            all_history.setdefault(new_sid, [])
            save_all_history(all_history)
            st.rerun()

        st.caption("Your chats")

        for sid in list(all_history.keys())[::-1]:
            msgs = all_history[sid]
            if not msgs:
                continue

            title = get_chat_title(msgs)
            is_selected = (sid == st.session_state["session_id"])

            c1, c2 = st.columns([1, 0.14], gap="xxsmall", vertical_alignment="center")

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

        # Simple logout
        if st.button("🚪 Log out", use_container_width=True):
            st.session_state["authentication_status"] = None
            st.session_state["username"] = None
            st.session_state["name"] = None
            st.rerun()

    # MAIN CONTENT
    st.title("⚖️ LegalGPT")
    st.caption("Indian Evidence Act • Production RAG System")

    if show_settings:
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        col_close, _ = st.columns([0.1, 1])
        with col_close:
            if st.button("✖"):
                st.query_params.clear()
                st.rerun()

        if st.button("🔄 Rebuild Index", use_container_width=True):
            with st.spinner("🔄 Re-indexing..."):
                docs = load_documents()
                if docs:
                    chunks = split_documents(docs)
                    vsm = VectorStoreManager()
                    vsm.add_documents(chunks)
                    st.success(f"✅ Indexed {len(chunks)} chunks.")

        if st.button("🗑️ Clear History", use_container_width=True):
            save_all_history({})
            st.session_state["messages"] = []
            st.session_state["session_id"] = str(uuid.uuid4())
            all_history = {st.session_state["session_id"]: []}
            save_all_history(all_history)
            st.rerun()

        st.markdown("---")

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
            placeholder.markdown(answer + "\\n\\n📚 *Powered by LegalRAG Pipeline*")

        st.session_state["messages"].append({"role": "assistant", "content": answer})

        all_history = load_all_history()
        all_history[st.session_state["session_id"]] = st.session_state["messages"]
        save_all_history(all_history)
        st.rerun()


if __name__ == "__main__":
    run_streamlit_app()
