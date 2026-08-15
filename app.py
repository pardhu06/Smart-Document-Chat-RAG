import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from huggingface_hub import InferenceClient

import matplotlib.pyplot as plt
from wordcloud import WordCloud


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Document Chat",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# HUGGING FACE API KEY
# ============================================================

try:
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
except Exception:
    st.error(
        "❌ Hugging Face API key not found.\n\n"
        "Create `.streamlit/secrets.toml` and add:\n\n"
        'HUGGINGFACE_API_KEY = "your_token_here"'
    )
    st.stop()


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=HUGGINGFACE_API_KEY
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #0f2027,
            #203a43,
            #2c5364
        );
    }

    .chat-user {
        background: linear-gradient(
            135deg,
            #4facfe,
            #00f2fe
        );

        color: black;
        padding: 12px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: right;
        font-weight: 500;
    }

    .chat-bot {
        background: rgba(255, 255, 255, 0.10);
        color: white;
        padding: 12px;
        border-radius: 15px;
        margin: 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "all_text" not in st.session_state:
    st.session_state.all_text = ""

if "document_name" not in st.session_state:
    st.session_state.document_name = ""


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload PDF",
    type=["pdf"]
)


# ============================================================
# CLEAR CHAT
# ============================================================

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.chat_history = []

    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Smart Document Chat")

st.write(
    "Chat with your PDF using Retrieval-Augmented Generation (RAG)."
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    # Process only if a new PDF is uploaded
    if (
        st.session_state.vectorstore is None
        or st.session_state.document_name != uploaded_file.name
    ):

        st.session_state.chat_history = []

        st.session_state.vectorstore = None

        st.session_state.all_text = ""

        st.session_state.document_name = uploaded_file.name

        # Save uploaded PDF temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("📄 Processing document..."):

            try:

                # Load PDF
                loader = PyPDFLoader("temp.pdf")

                documents = loader.load()

                # Split document
                splitter = CharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                docs = splitter.split_documents(documents)

                # Save text for WordCloud
                st.session_state.all_text = " ".join(
                    [doc.page_content for doc in docs]
                )

                # Create embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Create FAISS vector database
                st.session_state.vectorstore = (
                    FAISS.from_documents(
                        docs,
                        embeddings
                    )
                )

                st.sidebar.success(
                    "✅ Document ready!"
                )

            except Exception as e:

                st.error(
                    f"❌ Error while processing PDF:\n\n{e}"
                )

                st.stop()


# ============================================================
# WORD CLOUD
# ============================================================

if st.session_state.all_text:

    st.subheader("☁️ Document Insights")

    try:

        wordcloud = WordCloud(
            width=900,
            height=350,
            background_color="black"
        ).generate(
            st.session_state.all_text
        )

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        ax.imshow(
            wordcloud,
            interpolation="bilinear"
        )

        ax.axis("off")

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    except Exception as e:

        st.warning(
            f"Could not generate word cloud: {e}"
        )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask something from your document..."
)


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

if query:

    # Check whether document exists
    if st.session_state.vectorstore is None:

        st.warning(
            "📂 Please upload a PDF document first."
        )

    else:

        # Add user message
        st.session_state.chat_history.append(
            ("user", query)
        )

        with st.spinner("🤖 Thinking..."):

            try:

                # Retrieve relevant documents
                retrieved_docs = (
                    st.session_state.vectorstore
                    .similarity_search(
                        query,
                        k=3
                    )
                )

                # Create context
                context = "\n\n".join(
                    [
                        doc.page_content
                        for doc in retrieved_docs
                    ]
                )

                # RAG prompt
                prompt = f"""
You are a strict document assistant.

Answer the question ONLY using the information
provided in the context below.

If the answer cannot be found in the context,
say exactly:

"Answer not found in document."

Do not make up information.

Context:
{context}

Question:
{query}

Answer:
"""

                # Call Hugging Face
                response = client.chat_completion(

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    max_tokens=300,

                    temperature=0.2
                )

                answer = (
                    response.choices[0]
                    .message["content"]
                )

                # Save answer
                st.session_state.chat_history.append(
                    ("bot", answer)
                )

            except Exception as e:

                error_message = str(e)

                if "503" in error_message:

                    answer = (
                        "⚠️ The AI model is temporarily "
                        "unavailable because the service "
                        "is currently at capacity. "
                        "Please try again in a few moments."
                    )

                elif "401" in error_message:

                    answer = (
                        "❌ Hugging Face authentication "
                        "failed. Please check your API key "
                        "in `.streamlit/secrets.toml`."
                    )

                else:

                    answer = (
                        "❌ Error while generating answer:\n\n"
                        + error_message
                    )

                st.session_state.chat_history.append(
                    ("bot", answer)
                )


# ============================================================
# DISPLAY CHAT
# ============================================================

for role, msg in st.session_state.chat_history:

    if role == "user":

        st.markdown(
            f"""
            <div class="chat-user">
                🧑 {msg}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="chat-bot">
                🤖 {msg}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Smart Document Chat • RAG • LangChain • FAISS • Hugging Face"
)