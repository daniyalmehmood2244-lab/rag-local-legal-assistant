# ============================== #
# 📘 Import Required Libraries   #
# ============================== #

import os                      # For working with files and folders
import hashlib                 # To create unique file hashes (for identifying files)
import sqlite3                 # Lightweight local database (to store users and logs)
import bcrypt                  # For hashing passwords securely
import streamlit as st         # For creating the web interface
import pickle                  # To save and load FAISS index files
from datetime import datetime  # For timestamps in logs

# LangChain tools for document retrieval and Q&A
from langchain_community.document_loaders import PyPDFLoader  # Reads PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into chunks
from langchain_community.vectorstores import FAISS  # Fast search storage for embeddings
from langchain_community.embeddings import OllamaEmbeddings  # Creates embeddings using Ollama model
from langchain.chains import ConversationalRetrievalChain  # For Q&A chat
from langchain_community.llms import Ollama  # Connects to a local LLM (Ollama)

# ============================== #
# ⚙️ Database Setup              #
# ============================== #

DB_FILE = "app_data.db"  # SQLite database file name

def init_db():
    """Creates database tables for users and logs if they don’t already exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Create users table (to store usernames, hashed passwords, and roles)
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash BLOB,
            role TEXT
        )
    ''')

    # Create logs table (to store actions like login, uploads, deletions, etc.)
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            action TEXT,
            details TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def add_user(username, password, role="client"):
    """Add or update a user with a hashed password."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())  # Encrypt password
    c.execute("INSERT OR REPLACE INTO users (username, password_hash, role) VALUES (?,?,?)",
              (username, hashed, role))  # Add or update user
    conn.commit()
    conn.close()

def delete_user(username):
    """Delete a user from the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()

def get_all_users():
    """Get all registered users (username + role)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT username, role FROM users")
    users = c.fetchall()
    conn.close()
    return users

def check_login(username, password):
    """Check if username and password match."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password_hash, role FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()

    # Compare entered password with stored hash
    if row:
        stored_hash, role = row
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return True, role
    return False, None

def log_action(username, action, details):
    """Record user actions in logs (e.g., login, upload, question asked)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO logs (username, action, details) VALUES (?,?,?)",
              (username, action, details))
    conn.commit()
    conn.close()

def get_logs():
    """Fetch all log entries (most recent first)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT username, action, details, timestamp FROM logs ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# Create DB and ensure admin/client users exist
init_db()
add_user("admin", "admin123", "admin")
add_user("client", "client123", "client")

# ============================== #
# 🧾 File Handling & Indexing     #
# ============================== #

def get_file_hash(file):
    """Generate a unique hash for each file to avoid duplicates."""
    hasher = hashlib.md5()
    file.seek(0)
    hasher.update(file.read())
    file.seek(0)
    return hasher.hexdigest()

def save_upload_and_hash(uploaded_file):
    """Save uploaded file to disk and return file path and hash."""
    file_hash = get_file_hash(uploaded_file)
    os.makedirs("uploaded_files", exist_ok=True)  # Ensure folder exists
    save_path = f"uploaded_files/{file_hash}_{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path, file_hash

def build_or_load_index(pdf_paths, embed_model="mxbai-embed-large", chunk_size=1000, chunk_overlap=200):
    """Create or load a FAISS index for document retrieval."""
    index_dir = "vectorstores"
    os.makedirs(index_dir, exist_ok=True)
    faiss_file = os.path.join(index_dir, "combined_index.pkl")

    # If an index already exists, load it to save time
    if os.path.exists(faiss_file):
        with open(faiss_file, "rb") as f:
            return pickle.load(f)

    # Otherwise, create new embeddings from PDFs
    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)  # Read PDF
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_docs.extend(splitter.split_documents(docs))  # Break into smaller text pieces

    embeddings = OllamaEmbeddings(model=embed_model)
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    # Save FAISS index for reuse
    with open(faiss_file, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore

def load_existing_index():
    """Load saved FAISS index (for client to use)."""
    faiss_file = os.path.join("vectorstores", "combined_index.pkl")
    if os.path.exists(faiss_file):
        with open(faiss_file, "rb") as f:
            return pickle.load(f)
    return None

def clear_all_data():
    """Completely remove all uploaded files and indexes (admin only)."""
    upload_dir = "uploaded_files"
    vector_dir = "vectorstores"
    removed_files = []

    # Delete all uploaded files
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            os.remove(os.path.join(upload_dir, f))
            removed_files.append(f)

    # Delete FAISS index files
    if os.path.exists(vector_dir):
        for f in os.listdir(vector_dir):
            os.remove(os.path.join(vector_dir, f))
            removed_files.append(f)

    return removed_files

def make_chain(vectorstore, llm_model="llama3"):
    """Create a conversational Q&A chain using a local LLM and FAISS retriever."""
    llm = Ollama(model=llm_model)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Retrieve top 5 relevant chunks
        return_source_documents=True
    )

# ============================== #
# 💻 Streamlit UI                #
# ============================== #

# Configure Streamlit web app layout and title
st.set_page_config(page_title="RAG-based AI Legal Assistant:", layout="wide")
st.title("RAG-based AI Legal Assistant using LangChain, FAISS and Ollama")

# --- Session Setup ---
# Used to track login state between pages
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None

# --- LOGIN SCREEN ---
if not st.session_state.logged_in:
    st.subheader("Login")

    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Login button logic
    if st.button("Login"):
        ok, role = check_login(username, password)
        if ok:
            # Store user info in session
            st.session_state.logged_in = True
            st.session_state.role = role
            st.session_state.username = username
            log_action(username, "login", "User logged in")
            st.rerun()  # Refresh UI after login
        else:
            st.error("❌ Invalid credentials")

# --- MAIN APP ---
else:
    # Show who’s logged in on sidebar
    st.sidebar.write(f"👤 Logged in as **{st.session_state.username} ({st.session_state.role})**")

    # Logout button
    if st.sidebar.button("🚪 Logout"):
        log_action(st.session_state.username, "logout", "User logged out")
        st.session_state.clear()  # Clear session
        st.rerun()  # Go back to login page

    # --- ADMIN PANEL ---
    if st.session_state.role == "admin":
        st.header("📂 Admin Panel")

        # Upload PDFs section
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            pdf_paths = []
            for f in uploaded_files:
                pdf_path, _ = save_upload_and_hash(f)
                pdf_paths.append(pdf_path)
                log_action(st.session_state.username, "upload", f"Uploaded {f.name}")

            # Build FAISS index for uploaded documents
            with st.spinner("🔧 Building FAISS index..."):
                try:
                    vectorstore = build_or_load_index(pdf_paths)
                    st.session_state.chain = make_chain(vectorstore)
                    st.success("✅ Index built successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Button to clear all uploaded data
        if st.button("🧹 Clear All Uploaded Documents"):
            removed = clear_all_data()
            if removed:
                st.warning(f"🧾 Cleared {len(removed)} items.")
                log_action(st.session_state.username, "clear", f"Cleared {len(removed)} files.")
            else:
                st.info("No data found to clear.")

        # --- 🧑‍💼 Manage Users ---
        st.subheader("🧑‍💼 Manage Users")

        # Add or update user section
        with st.expander("Add or Update User"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["client", "admin"])
            if st.button("Save User"):
                if new_username and new_password:
                    add_user(new_username, new_password, new_role)
                    st.success(f"✅ User '{new_username}' saved ({new_role})")
                    log_action(st.session_state.username, "user_add", f"Added {new_username} ({new_role})")
                else:
                    st.warning("Please enter both username and password.")

        # Delete user section
        with st.expander("Delete User"):
            users = get_all_users()
            usernames = [u[0] for u in users if u[0] != "admin"]  # Prevent deleting admin
            if usernames:
                user_to_delete = st.selectbox("Select user to delete", usernames)
                if st.button("Delete User"):
                    delete_user(user_to_delete)
                    st.warning(f"🗑️ Deleted user '{user_to_delete}'")
                    log_action(st.session_state.username, "user_delete", f"Deleted {user_to_delete}")
            else:
                st.info("No users to delete.")

        # View system logs
        with st.expander("📜 View Logs"):
            logs = get_logs()
            for row in logs:
                st.write(f"{row[3]} — **{row[0]}** {row[1]}: {row[2]}")

    # --- CLIENT PANEL / CHAT SECTION ---
    if "chain" not in st.session_state:
        vectorstore = load_existing_index()
        if vectorstore:
            st.session_state.chain = make_chain(vectorstore)
            st.success("✅ Loaded saved knowledge base.")
        else:
            st.info("No indexed documents found. (Admin must upload PDFs first.)")

    # Chat section (visible to both admin and clients)
    st.header("💬 Ask Questions About the Documents")

    if "chain" in st.session_state:
        # Sidebar option to clear chat
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat cleared!")

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Show past messages
        for user_msg, bot_msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                st.markdown(bot_msg)

        # Input new question
        user_question = st.chat_input("💬 Ask your question...")
        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    # Send query to model and get answer
                    result = st.session_state.chain(
                        {"question": user_question, "chat_history": st.session_state.chat_history}
                    )
                    answer = result["answer"]
                    st.markdown(answer)

            # Save chat and log action
            st.session_state.chat_history.append((user_question, answer))
            log_action(st.session_state.username, "question", user_question)
    else:
        st.info("No active knowledge base found. Please upload PDFs as Admin first.")
