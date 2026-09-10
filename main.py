import os
import math
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv


# ==========================================
# SETUP
# ==========================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Study Assistant")
st.write(
    "Your AI-powered study companion for questions, "
    "documents, summaries, quizzes, and flashcards."
)


# ==========================================
# SESSION STATE
# ==========================================

if "general_messages" not in st.session_state:
    st.session_state.general_messages = []

if "pdf_messages" not in st.session_state:
    st.session_state.pdf_messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None


# ==========================================
# RAG FUNCTIONS
# ==========================================

def create_chunks(reader):

    chunks = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if not text:
            continue

        chunk_size = 1200
        overlap = 200

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "page": page_number
            })

            start += chunk_size - overlap

    return chunks


def create_embeddings(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [
        item.embedding
        for item in response.data
    ]


def cosine_similarity(vector1, vector2):

    dot_product = sum(
        a * b
        for a, b in zip(vector1, vector2)
    )

    magnitude1 = math.sqrt(
        sum(a * a for a in vector1)
    )

    magnitude2 = math.sqrt(
        sum(b * b for b in vector2)
    )

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (
        magnitude1 * magnitude2
    )


def retrieve_relevant_chunks(
    question,
    top_k=4
):

    question_embedding = (
        client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        )
        .data[0]
        .embedding
    )

    scores = []

    for index, embedding in enumerate(
        st.session_state.embeddings
    ):

        score = cosine_similarity(
            question_embedding,
            embedding
        )

        scores.append(
            (score, index)
        )

    scores.sort(reverse=True)

    best_chunks = []

    for score, index in scores[:top_k]:

        best_chunks.append(
            st.session_state.chunks[index]
        )

    return best_chunks


def ask_pdf(question):

    relevant_chunks = (
        retrieve_relevant_chunks(question)
    )

    context = ""
    pages = []

    for chunk in relevant_chunks:

        context += (
            f"\n--- Page {chunk['page']} ---\n"
            f"{chunk['text']}\n"
        )

        pages.append(
            chunk["page"]
        )

    prompt = f"""
You are an AI Study Assistant.

Answer the student's question using ONLY
the provided study material.

If the answer cannot be found in the material,
say that you could not find enough information
in the uploaded document.

Explain the answer clearly.

STUDY MATERIAL:

{context}

STUDENT QUESTION:

{question}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return (
        response.output_text,
        sorted(set(pages))
    )


# ==========================================
# TABS
# ==========================================

tutor_tab, pdf_tab = st.tabs([
    "💬 AI Tutor",
    "📄 Study My PDF"
])


# ==========================================
# AI TUTOR
# ==========================================

with tutor_tab:

    st.header("💬 AI Tutor")

    st.write(
        "Ask questions about programming, "
        "computer science, math, or other "
        "subjects."
    )

    if st.button(
        "Clear Tutor Chat",
        key="clear_tutor"
    ):

        st.session_state.general_messages = []

        st.rerun()


    # Display previous messages

    for message in (
        st.session_state.general_messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # Chat input

    tutor_question = st.chat_input(
        "Ask your AI Tutor anything...",
        key="tutor_input"
    )

    if tutor_question:

        st.session_state.general_messages.append({
            "role": "user",
            "content": tutor_question
        })

        with st.chat_message("user"):

            st.write(
                tutor_question
            )


        # Build conversation history

        conversation = ""

        for message in (
            st.session_state.general_messages[-10:]
        ):

            conversation += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )


        prompt = f"""
You are a helpful AI Tutor for a college student.

Explain concepts clearly and step-by-step.

For programming questions, explain code in
beginner-friendly language and provide examples
when useful.

Conversation:

{conversation}

Respond to the student's latest message.
"""


        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                response = (
                    client.responses.create(
                        model="gpt-5.6-luna",
                        input=prompt
                    )
                )

                answer = (
                    response.output_text
                )

            st.write(answer)


        st.session_state.general_messages.append({
            "role": "assistant",
            "content": answer
        })


# ==========================================
# PDF STUDY MODE
# ==========================================

with pdf_tab:

    st.header("📄 Study My PDF")

    st.write(
        "Upload your class notes, slides, "
        "or readings and study them with AI."
    )


    uploaded_file = st.file_uploader(
        "Upload your study PDF",
        type=["pdf"]
    )


    if uploaded_file is not None:

        # Process new PDF

        if (
            st.session_state.current_file
            != uploaded_file.name
        ):

            reader = PdfReader(
                uploaded_file
            )

            with st.spinner(
                "Reading and analyzing your PDF..."
            ):

                chunks = create_chunks(
                    reader
                )

                if len(chunks) == 0:

                    st.error(
                        "I couldn't extract text "
                        "from this PDF."
                    )

                    st.stop()


                embeddings = (
                    create_embeddings(chunks)
                )


                st.session_state.chunks = (
                    chunks
                )

                st.session_state.embeddings = (
                    embeddings
                )

                st.session_state.current_file = (
                    uploaded_file.name
                )

                st.session_state.pdf_messages = []


            st.success(
                "PDF processed successfully!"
            )


        st.info(
            f"📖 Currently studying: "
            f"{uploaded_file.name}"
        )


        # Clear PDF chat

        if st.button(
            "Clear PDF Chat",
            key="clear_pdf"
        ):

            st.session_state.pdf_messages = []

            st.rerun()


        # Display PDF conversation

        for message in (
            st.session_state.pdf_messages
        ):

            with st.chat_message(
                message["role"]
            ):

                st.write(
                    message["content"]
                )

                if "pages" in message:

                    st.caption(
                        "Relevant PDF pages: "
                        + ", ".join(
                            str(page)
                            for page
                            in message["pages"]
                        )
                    )


        # PDF question

        pdf_question = st.chat_input(
            "Ask something about your PDF...",
            key="pdf_input"
        )


        if pdf_question:

            st.session_state.pdf_messages.append({
                "role": "user",
                "content": pdf_question
            })


            with st.chat_message("user"):

                st.write(
                    pdf_question
                )


            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching your document..."
                ):

                    answer, pages = (
                        ask_pdf(pdf_question)
                    )


                st.write(answer)

                st.caption(
                    "Relevant PDF pages: "
                    + ", ".join(
                        str(page)
                        for page in pages
                    )
                )


            st.session_state.pdf_messages.append({
                "role": "assistant",
                "content": answer,
                "pages": pages
            })


        # ==================================
        # STUDY TOOLS
        # ==================================

        st.divider()

        st.subheader(
            "🧠 Study Tools"
        )


        col1, col2, col3 = st.columns(3)


        combined_text = "\n".join(
            chunk["text"]
            for chunk
            in st.session_state.chunks
        )


        # SUMMARY

        with col1:

            if st.button(
                "📝 Summary",
                use_container_width=True
            ):

                prompt = f"""
Create a clear study summary of the
following material.

Include:

- Main ideas
- Important concepts
- Key definitions
- Important facts

Make it useful for exam review.

MATERIAL:

{combined_text}
"""


                with st.spinner(
                    "Creating summary..."
                ):

                    response = (
                        client.responses.create(
                            model="gpt-5.6-luna",
                            input=prompt
                        )
                    )


                st.subheader(
                    "📝 Study Summary"
                )

                st.write(
                    response.output_text
                )


        # QUIZ

        with col2:

            if st.button(
                "❓ Quiz",
                use_container_width=True
            ):

                prompt = f"""
Create a 5-question multiple-choice
quiz from this study material.

Each question should have:

A)
B)
C)
D)

Put the correct answer after each
question.

MATERIAL:

{combined_text}
"""


                with st.spinner(
                    "Creating quiz..."
                ):

                    response = (
                        client.responses.create(
                            model="gpt-5.6-luna",
                            input=prompt
                        )
                    )


                st.subheader(
                    "❓ Practice Quiz"
                )

                st.write(
                    response.output_text
                )


        # FLASHCARDS

        with col3:

            if st.button(
                "🗂️ Flashcards",
                use_container_width=True
            ):

                prompt = f"""
Create 10 study flashcards from
the following material.

Format each as:

Question:
Answer:

Keep them concise and useful.

MATERIAL:

{combined_text}
"""


                with st.spinner(
                    "Creating flashcards..."
                ):

                    response = (
                        client.responses.create(
                            model="gpt-5.6-luna",
                            input=prompt
                        )
                    )


                st.subheader(
                    "🗂️ Flashcards"
                )

                st.write(
                    response.output_text
                )


    else:

        st.info(
            "Upload a PDF to use document study mode."
        )