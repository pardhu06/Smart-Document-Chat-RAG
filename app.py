import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from huggingface_hub import InferenceClient

import matplotlib.pyplot as plt
from wordcloud import WordCloud


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Document Chat",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# HUGGING FACE API
# ============================================================

try:
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
except Exception:
    st.error(
        "❌ Hugging Face API key not found.\n\n"
        "Create .streamlit/secrets.toml and add:\n\n"
        "HUGGINGFACE_API_KEY = \"your_token_here\""
    )
    st.stop()


client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
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
        padding: 15px;
        border-radius: 15px;
        margin: 12px 0;
        text-align: right;
        font-size: 16px;
    }

    .chat-bot {
        background: rgba(255,255,255,0.10);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 12px 0;
        font-size: 16px;
        line-height: 1.6;
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

if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Controls")

st.sidebar.write(
    "Upload a PDF and ask questions about it."
)

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload PDF",
    type=["pdf"]
)


# ============================================================
# CLEAR CHAT
# ============================================================

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.chat_history = []
    st.session_state.retrieved_docs = []

    st.rerun()


# ============================================================
# RESET DOCUMENT
# ============================================================

if st.sidebar.button("🔄 Reset Document"):

    st.session_state.vectorstore = None
    st.session_state.all_text = ""
    st.session_state.retrieved_docs = []
    st.session_state.file_name = None
    st.session_state.chat_history = []

    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Smart Document Chat")

st.write(
    "Chat with your PDF using "
    "**Retrieval-Augmented Generation (RAG)**."
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    if uploaded_file.name != st.session_state.file_name:

        # ----------------------------------------------------
        # Save uploaded PDF
        # ----------------------------------------------------

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())


        # ----------------------------------------------------
        # Reset previous document
        # ----------------------------------------------------

        st.session_state.vectorstore = None
        st.session_state.all_text = ""
        st.session_state.retrieved_docs = []
        st.session_state.chat_history = []

        st.session_state.file_name = uploaded_file.name


        # ----------------------------------------------------
        # Process PDF
        # ----------------------------------------------------

        try:

            with st.spinner("📄 Processing your PDF..."):

                # Load PDF
                loader = PyPDFLoader("temp.pdf")

                documents = loader.load()

                if not documents:
                    st.error(
                        "❌ Could not extract text from PDF."
                    )
                    st.stop()


                # ------------------------------------------------
                # Split document into chunks
                # ------------------------------------------------

                splitter = CharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                docs = splitter.split_documents(
                    documents
                )


                # ------------------------------------------------
                # Save complete text
                # ------------------------------------------------

                st.session_state.all_text = " ".join(
                    doc.page_content
                    for doc in docs
                )


                # ------------------------------------------------
                # Create embeddings
                # ------------------------------------------------

                embeddings = HuggingFaceEmbeddings(
                    model_name=
                    "sentence-transformers/all-MiniLM-L6-v2"
                )


                # ------------------------------------------------
                # Create FAISS vector database
                # ------------------------------------------------

                st.session_state.vectorstore = (
                    FAISS.from_documents(
                        docs,
                        embeddings
                    )
                )


            st.sidebar.success(
                "✅ Document ready!"
            )

            st.success(
                f"📄 {uploaded_file.name} "
                "processed successfully!"
            )


        except Exception as e:

            st.error(
                f"❌ Error while processing PDF:\n\n{str(e)}"
            )

            st.session_state.vectorstore = None


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.vectorstore is not None:

    st.subheader("📊 Document Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Document",
            st.session_state.file_name
        )

    with col2:

        word_count = len(
            st.session_state.all_text.split()
        )

        st.metric(
            "📝 Words",
            word_count
        )

    with col3:

        character_count = len(
            st.session_state.all_text
        )

        st.metric(
            "🔤 Characters",
            character_count
        )


# ============================================================
# WORD CLOUD
# ============================================================

if st.session_state.all_text.strip():

    st.subheader("☁️ Document Insights")

    try:

        wordcloud = WordCloud(
            width=900,
            height=350,
            background_color="black",
            max_words=100,
            collocations=False
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


    except ValueError:

        st.warning(
            "⚠️ Not enough text to generate word cloud."
        )


# ============================================================
# ASK QUESTIONS
# ============================================================

st.subheader("💬 Ask Questions")


if st.session_state.vectorstore is None:

    st.info(
        "📂 Upload a PDF from the sidebar "
        "to start asking questions."
    )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask something from your document..."
)


# ============================================================
# RAG PIPELINE
# ============================================================

if query:

    if st.session_state.vectorstore is None:

        st.warning(
            "⚠️ Please upload a PDF first."
        )

    else:

        # ----------------------------------------------------
        # Save user question
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            ("user", query)
        )


        try:

            with st.spinner(
                "🔎 Searching your document..."
            ):

                # ------------------------------------------------
                # Similarity search
                # ------------------------------------------------

                retrieved_docs = (
                    st.session_state.vectorstore
                    .similarity_search(
                        query,
                        k=3
                    )
                )


                # ------------------------------------------------
                # Save retrieved documents
                # ------------------------------------------------

                st.session_state.retrieved_docs = (
                    retrieved_docs
                )


                # ------------------------------------------------
                # Create context
                # ------------------------------------------------

                context = "\n\n".join(
                    doc.page_content
                    for doc in retrieved_docs
                )


                # ------------------------------------------------
                # Prompt
                # ------------------------------------------------

                prompt = f"""
You are a strict document assistant.

Answer the user's question ONLY using the
information contained in the provided context.

Rules:

1. Do not use outside knowledge.
2. Do not make up information.
3. Give a clear and simple answer.
4. If the answer cannot be found in the context,
   say exactly:

Answer not found in document.

Context:
{context}

Question:
{query}

Answer:
"""


                # ------------------------------------------------
                # Hugging Face LLM
                # ------------------------------------------------

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


                # ------------------------------------------------
                # Get answer
                # ------------------------------------------------

                answer = (
                    response
                    .choices[0]
                    .message["content"]
                )


            # ----------------------------------------------------
            # Save assistant answer
            # ----------------------------------------------------

            st.session_state.chat_history.append(
                ("bot", answer)
            )


        except Exception as e:

            error_message = (
                "❌ Error while generating answer:\n\n"
                + str(e)
            )

            st.session_state.chat_history.append(
                ("bot", error_message)
            )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for role, msg in st.session_state.chat_history:

    if role == "user":

        st.markdown(
            f"""
            <div class="chat-user">
                👤 <b>You:</b> {msg}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="chat-bot">
                🤖 <b>Assistant:</b>
                <br><br>
                {msg}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# RETRIEVED CONTEXT
# ============================================================

