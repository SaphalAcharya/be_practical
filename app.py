import os
import streamlit as st
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Official Google GenAI SDK
from google import genai
from google.genai import types


# Load environment variables (API Key)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Initialize Gemini Client
if not GEMINI_API_KEY:
    st.error("🚨 GEMINI_API_KEY missing. Please add it to your .env file.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)


# ==============================================================================
# CUSTOM EMBEDDING FUNCTION & UI SHELL
# ==============================================================================

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom ChromaDB embedding function using Gemini Embedding.
    """

    def __call__(self, input: Documents) -> Embeddings:

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=input
        )

        return [emb.values for emb in response.embeddings]

# Streamlit Page Config
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Secure Enterprise Document RAG")

st.markdown(
    "Upload a PDF document and ask questions. The AI will cite its sources."
)


# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection" not in st.session_state:
    st.session_state.collection = None


# ==============================================================================
# SIDEBAR: DOCUMENT INGESTION
# ==============================================================================

with st.sidebar:
    st.header("📄 Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type="pdf"
    )


# ==============================================================================
# PDF PROCESSING AND CHROMADB INGESTION
# ==============================================================================

if (
    uploaded_file is not None
    and st.session_state.collection is None
):

    with st.spinner("Parsing and chunking document..."):

        # 1. Read PDF
        reader = PdfReader(uploaded_file)

        raw_text = "".join(
            [
                page.extract_text() + "\n"
                for page in reader.pages
            ]
        )

        # 2. Chunk Text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_text(raw_text)

        # 3. Initialize Ephemeral ChromaDB
        chroma_client = chromadb.Client()

        collection = chroma_client.create_collection(
          name="temp_doc_db",
          embedding_function=GeminiEmbeddingFunction()
)

        # 4. Ingest
        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": uploaded_file.name,
                "chunk": i
            }
            for i in range(len(chunks))
        ]

        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

        st.session_state.collection = collection

        st.success(
            f"✅ Indexed {len(chunks)} chunks successfully!"
        )


# ==============================================================================
# DISPLAY CHAT HISTORY
# ==============================================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if "sources" in msg:

            with st.expander("View Retrieved Sources"):

                for source in msg["sources"]:
                    st.info(source)


# ==============================================================================
# 🎓 STUDENT LAB WORKSPACE
# RETRIEVAL, GUARDRAILS & GENERATION
# ==============================================================================

# TODO: Continue with Task 1, Task 2 and Task 3 here.













# ==============================================================================
# STUDENT LAB WORKSPACE
# RETRIEVAL, GUARDRAILS & GENERATION
# ==============================================================================


# ==============================================================================
# TASK 1: RETRIEVE CONTEXT FROM VECTOR DATABASE
# ==============================================================================

prompt = st.chat_input(
    "Ask a question about your uploaded document..."
)


if prompt:

    # --------------------------------------------------------------------------
    # DISPLAY USER QUESTION
    # --------------------------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(prompt)


    # --------------------------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # --------------------------------------------------------------------------
    # CHECK IF DOCUMENT IS UPLOADED
    # --------------------------------------------------------------------------

    if st.session_state.collection is None:

        final_answer = (
            "⚠️ Please upload a PDF document first."
        )

        with st.chat_message("assistant"):

            st.markdown(final_answer)


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_answer
            }
        )


    else:

        # ----------------------------------------------------------------------
        # TODO 1: QUERY CHROMADB
        # ----------------------------------------------------------------------

        results = st.session_state.collection.query(
            query_texts=[prompt],
            n_results=3
        )


        # ----------------------------------------------------------------------
        # GET RETRIEVED DOCUMENTS
        # ----------------------------------------------------------------------

        retrieved_documents = results["documents"][0]

        retrieved_metadatas = results["metadatas"][0]


        # ----------------------------------------------------------------------
        # CREATE context_text AND source_list
        # ----------------------------------------------------------------------

        context_parts = []

        source_list = []


        for document, metadata in zip(
            retrieved_documents,
            retrieved_metadatas
        ):

            chunk_number = metadata["chunk"]


            # Add chunk to context
            context_parts.append(
                f"[Chunk {chunk_number}]\n{document}"
            )


            # Add chunk to source list
            source_list.append(
                f"Chunk {chunk_number}:\n{document}"
            )


        context_text = "\n\n".join(context_parts)


        # ======================================================================
        # TASK 2: BUILD THE GUARDRAILED PROMPT
        # ======================================================================

        system_instruction = """
You are a strict Enterprise Q&A Assistant.

Guardrail 1:
Answer ONLY using the information provided in the context.

Guardrail 2:
If the answer is not contained in the context, explicitly state:

"I cannot answer this based on the provided documents."

Do NOT hallucinate.
Do NOT use outside knowledge.

Guardrail 3:
When making claims, cite the relevant chunk number inline.

For example:
[Chunk 1]

If multiple chunks support your answer, cite all relevant chunks.
"""


        # ----------------------------------------------------------------------
        # COMBINE CONTEXT AND USER QUESTION
        # ----------------------------------------------------------------------

        user_payload = f"""
CONTEXT FROM THE PROVIDED DOCUMENT:

{context_text}


USER QUESTION:

{prompt}


INSTRUCTIONS:

Answer the user's question using ONLY the context provided above.

If the answer is not contained in the context, say:

"I cannot answer this based on the provided documents."

Do NOT use outside knowledge.
Do NOT hallucinate.
"""


        # ======================================================================
        # TASK 3: EXECUTE LLM GENERATION
        # ======================================================================

        try:

            with st.chat_message("assistant"):

                with st.spinner("Generating answer..."):

                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=user_payload,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.0
                        )
                    )


                    # ----------------------------------------------------------
                    # EXTRACT GENERATED TEXT
                    # ----------------------------------------------------------

                    final_answer = response.text


                    # ----------------------------------------------------------
                    # DISPLAY GENERATED ANSWER
                    # ----------------------------------------------------------

                    st.markdown(final_answer)


                    # ----------------------------------------------------------
                    # DISPLAY RETRIEVED SOURCES
                    # ----------------------------------------------------------

                    with st.expander(
                        "View Retrieved Sources"
                    ):

                        for source in source_list:

                            st.info(source)


            # ------------------------------------------------------------------
            # SAVE ASSISTANT RESPONSE TO CHAT HISTORY
            # ------------------------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "sources": source_list
                }
            )


        # ======================================================================
        # ERROR HANDLING
        # ======================================================================

        except Exception as e:

            error_message = (
                f"❌ An error occurred while generating the answer:\n\n"
                f"{str(e)}"
            )


            with st.chat_message("assistant"):

                st.error(error_message)


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )
