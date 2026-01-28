import streamlit as st
import json
import uuid
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
import traceback
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # Legal-Rag-Assistent-/
sys.path.insert(0, str(ROOT_DIR))
# --- STANDARD LANGCHAIN IMPORTS ---
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
            if isinstance(data, list):
                return {}
            return data
        except Exception:
            return {}
    return {}


def save_all_history(all_history):
    HISTORY_FILE.write_text(json.dumps(all_history, ensure_ascii=False, indent=2), encoding="utf-8")


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
        page_title="Legal Research Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not CONFIG_PATH.exists():
        st.error("❌ config.yaml not found in project root.")
        st.stop()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader)

    config.setdefault("credentials", {}).setdefault("usernames", {})

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )

    # -------------------- AUTH --------------------
    name, authentication_status, username = authenticator.login(
        location="sidebar",
        fields={
            "Form name": "🔐 Login",
            "Username": "Username",
            "Password": "Password",
            "Login": "Login",
        },
    )

    if authentication_status in (None, False):
        st.sidebar.markdown("---")
        with st.sidebar.expander("📝 Create Account", expanded=False):
            with st.form("signup_form", clear_on_submit=True):
                new_fullname = st.text_input("Full Name")
                new_email = st.text_input("Email")
                new_user = st.text_input("Username")
                new_pass = st.text_input("Password", type="password")
                new_pass2 = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Create Account")

            if submitted:
                if not (new_fullname and new_email and new_user and new_pass and new_pass2):
                    st.error("All fields are required!")
                elif new_pass != new_pass2:
                    st.error("Passwords do not match!")
                elif new_user in config["credentials"]["usernames"]:
                    st.error("Username already exists!")
                else:
                    hashed = Hasher([new_pass]).generate()[0]
                    config["credentials"]["usernames"][new_user] = {
                        "name": new_fullname,
                        "email": new_email,
                        "password": hashed,
                    }
                    save_config(config)
                    st.success("✅ Account created! Please login.")
                    st.rerun()

    if authentication_status is False:
        st.error("❌ Username/password is incorrect")
        st.stop()
    if authentication_status is None:
        st.warning("👋 Please login (or sign up) to continue.")
        st.stop()

    # -------------------- CSS (Light Black Theme) --------------------
    st.markdown(
        """
        <style>
        /* Sidebar Background: ChatGPT-like Dark Grey (#171717) */
        [data-testid="stSidebar"] { background-color: #171717; }
        
        /* Main App Background: slightly lighter grey */
        .stApp { background-color: #212121; }

        div.stButton > button { text-align: left; border: none; background: transparent; color: #ececf1; }
        div.stButton > button:hover { background: #2f2f2f; color: white; }

        /* Sidebar bottom fixed container */
        .sidebar-bottom {
            position: sticky;
            bottom: 0;
            width: 100%;
            background: #171717;
            border-top: 1px solid #303030;
            padding: 10px 12px 12px 12px;
            z-index: 10;
        }

        .profile-row {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 8px;
            border-radius: 10px;
        }
        .profile-row:hover { background: #2f2f2f; }

        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            background: #7b4ec9;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
        }

        .pname { color: #fff; font-size: 14px; font-weight: 600; line-height: 1.1; }
        .pplan { color: #b4b4b4; font-size: 12px; line-height: 1.1; margin-top: 2px; }

        .settings-link {
            display: block;
            margin-top: 10px;
            padding: 11px 12px;
            border-radius: 12px;
            border: 1px solid #303030;
            color: #ececf1 !important;
            background: transparent;
            font-size: 14px;
            text-decoration: none !important;
            text-align: left;
        }
        .settings-link:hover { background: #2f2f2f; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------- SESSION --------------------
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []

    all_history = load_all_history()
    qp = st.query_params
    show_settings = (qp.get("menu") == "settings")

    # -------------------- SIDEBAR CONTENT --------------------
    with st.sidebar:
        if st.button("➕ New chat", use_container_width=True, type="secondary"):
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.rerun()

        st.caption("Your chats")
        for sid in list(all_history.keys())[::-1]:
            msgs = all_history[sid]
            if not msgs:
                continue
            title = get_chat_title(msgs)
            label = f"🟢 {title}" if sid == st.session_state["session_id"] else title
            if st.button(label, key=sid, use_container_width=True):
                st.session_state["session_id"] = sid
                st.session_state["messages"] = msgs
                st.rerun()

        # Spacer
        st.markdown("<div style='flex-grow: 1; height: 48vh;'></div>", unsafe_allow_html=True)

        if show_settings:
            st.markdown("---")
            c1, c2 = st.columns([1, 1])
            with c2:
                if st.button("✖ Close", use_container_width=True):
                    st.query_params.clear()
                    st.rerun()

            st.subheader("Settings")
            if st.button("Build Index", use_container_width=True):
                with st.spinner("Indexing..."):
                   
                    docs = load_documents()
                    if docs:
                        chunks = split_documents(docs)
                        vsm = VectorStoreManager()
                        vsm.add_documents(chunks)
                        st.success(f"✅ Indexed {len(chunks)} chunks.")
                    
            if st.button("🗑️ Clear History", use_container_width=True):
                save_all_history({})
                st.session_state["messages"] = []
                st.rerun()
            authenticator.logout("🚪 Log out", "sidebar")

        # --- BOTTOM PROFILE (STICKY) ---
        initials = (name[:2].upper() if name else "US")
        st.markdown(
            f"""
            <div class="sidebar-bottom">
              <div class="profile-row">
                <div class="avatar">{initials}</div>
                <div>
                  <div class="pname">{name}</div>
                  <div class="pplan">Free Plan</div>
                </div>
              </div>

              <a class="settings-link" href="?menu=settings" target="_self">⚙️ Settings</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------- MAIN CHAT --------------------
    st.title("⚖️ LegalGPT")
    st.caption("Indian Evidence Act • Production RAG System")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a legal question..."):
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
        all_history[st.session_state["session_id"]] = st.session_state["messages"]
        save_all_history(all_history)
        st.rerun()

if __name__ == "__main__":
    run_streamlit_app()
