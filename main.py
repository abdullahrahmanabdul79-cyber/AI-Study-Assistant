import os
import math
import json
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

if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []

if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "quiz_answered" not in st.session_state:
    st.session_state.quiz_answered = False

if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

if "flashcard_index" not in st.session_state:
    st.session_state.flashcard_index = 0

if "show_flashcard_answer" not in st.session_state:
    st.session_state.show_flashcard_answer = False


# ==========================================
# RAG FUNCTIONS
# ==========================================

def create_chunks(reader):

    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):

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


def retrieve_relevant_chunks(question, top_k=4):

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

    relevant_chunks = retrieve_relevant_chunks(question)

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

Explain the answer clearly and simply.

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
# MAIN TABS
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
        "Ask anything about programming, computer science, "
        "math, or other subjects."
    )

    if st.button(
        "Clear Tutor Chat",
        key="clear_tutor"
    ):

        st.session_state.general_messages = []

        st.rerun()


    # Display previous messages

    for message in st.session_state.general_messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


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


        conversation = ""

        for message in st.session_state.general_messages[-10:]:

            conversation += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )


        prompt = f"""
You are a helpful AI Tutor for a college student.

Explain concepts clearly and step-by-step.

For programming questions, use beginner-friendly
language and examples when useful.

Conversation:

{conversation}

Respond to the student's latest message.
"""


        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = client.responses.create(
                    model="gpt-5.6-luna",
                    input=prompt
                )

                answer = response.output_text


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
        "Upload your notes, slides, readings, "
        "or textbook chapters and study them with AI."
    )


    uploaded_file = st.file_uploader(
        "Upload your study PDF",
        type=["pdf"]
    )


    if uploaded_file is not None:

        # PROCESS NEW PDF
        if (
            st.session_state.current_file
            != uploaded_file.name
        ):

            reader = PdfReader(uploaded_file)


            with st.spinner(
                "Reading and analyzing your PDF..."
            ):

                chunks = create_chunks(reader)


                if len(chunks) == 0:

                    st.error(
                        "I couldn't extract readable "
                        "text from this PDF."
                    )

                    st.stop()


                embeddings = create_embeddings(
                    chunks
                )


                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings

                st.session_state.current_file = (
                    uploaded_file.name
                )

                st.session_state.pdf_messages = []

                st.session_state.quiz_questions = []
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answered = False

                st.session_state.flashcards = []
                st.session_state.flashcard_index = 0
                st.session_state.show_flashcard_answer = False


            st.success(
                "PDF processed successfully!"
            )


        st.info(
            f"📖 Currently studying: "
            f"{uploaded_file.name}"
        )


        if st.button(
            "Clear PDF Chat",
            key="clear_pdf"
        ):

            st.session_state.pdf_messages = []

            st.rerun()


        # DISPLAY PDF CHAT HISTORY
        for message in st.session_state.pdf_messages:

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
                            for page in message["pages"]
                        )
                    )


        # PDF CHAT INPUT
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

                    answer, pages = ask_pdf(
                        pdf_question
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

        st.subheader("🧠 Study Tools")


        combined_text = "\n".join(
            chunk["text"]
            for chunk in st.session_state.chunks
        )


        summary_tab, quiz_tab, flashcard_tab = st.tabs([
            "📝 Summary",
            "❓ Interactive Quiz",
            "🗂️ Flashcards"
        ])


        # ==================================
        # SUMMARY
        # ==================================

        with summary_tab:

            st.write(
                "Generate a study summary "
                "from your uploaded PDF."
            )


            if st.button(
                "Generate Summary",
                use_container_width=True
            ):

                prompt = f"""
Create a clear study summary from the material below.

Include:

- Main ideas
- Important concepts
- Key definitions
- Important facts

Make it useful for a college student
preparing for an exam.

STUDY MATERIAL:

{combined_text}
"""


                with st.spinner(
                    "Creating your study summary..."
                ):

                    response = client.responses.create(
                        model="gpt-5.6-luna",
                        input=prompt
                    )


                st.subheader(
                    "📝 Study Summary"
                )

                st.write(
                    response.output_text
                )


        # ==================================
        # INTERACTIVE QUIZ
        # ==================================

        with quiz_tab:

            st.write(
                "Generate a quiz and test yourself."
            )


            if not st.session_state.quiz_questions:

                if st.button(
                    "🎯 Start New Quiz",
                    use_container_width=True
                ):

                    prompt = f"""
Create exactly 5 multiple-choice questions
from the study material below.

Return ONLY valid JSON.

Use this exact structure:

[
    {{
        "question": "Question here",
        "options": [
            "Option A",
            "Option B",
            "Option C",
            "Option D"
        ],
        "answer": "Option B",
        "explanation": "Short explanation here"
    }}
]

Rules:

- Create exactly 5 questions.
- Each question must have exactly 4 options.
- The answer must exactly match one option.
- Return JSON only.
- Do not include markdown.
- Do not include ```json.

STUDY MATERIAL:

{combined_text}
"""


                    with st.spinner(
                        "Creating your quiz..."
                    ):

                        response = client.responses.create(
                            model="gpt-5.6-luna",
                            input=prompt
                        )


                    try:

                        quiz_data = json.loads(
                            response.output_text
                        )

                        st.session_state.quiz_questions = quiz_data
                        st.session_state.quiz_index = 0
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_answered = False

                        st.rerun()


                    except json.JSONDecodeError:

                        st.error(
                            "The quiz couldn't be created "
                            "correctly. Try again."
                        )


            else:

                total_questions = len(
                    st.session_state.quiz_questions
                )

                current_index = (
                    st.session_state.quiz_index
                )


                # QUIZ FINISHED
                if current_index >= total_questions:

                    st.success(
                        "🎉 Quiz Complete!"
                    )


                    score = (
                        st.session_state.quiz_score
                    )

                    percentage = int(
                        (
                            score
                            / total_questions
                        )
                        * 100
                    )


                    st.metric(
                        "Your Score",
                        f"{score}/{total_questions}"
                    )

                    st.progress(
                        score / total_questions
                    )

                    st.write(
                        f"### {percentage}%"
                    )


                    if percentage == 100:

                        st.success(
                            "Perfect score! Excellent work! 🏆"
                        )

                    elif percentage >= 80:

                        st.success(
                            "Great job! You understand "
                            "this material well."
                        )

                    elif percentage >= 60:

                        st.info(
                            "Good attempt. Review the "
                            "questions you missed."
                        )

                    else:

                        st.warning(
                            "Keep studying and try again."
                        )


                    if st.button(
                        "🔄 Create Another Quiz",
                        use_container_width=True
                    ):

                        st.session_state.quiz_questions = []
                        st.session_state.quiz_index = 0
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_answered = False

                        st.rerun()


                # CURRENT QUIZ QUESTION
                else:

                    question_data = (
                        st.session_state.quiz_questions[
                            current_index
                        ]
                    )


                    st.write(
                        f"### Question "
                        f"{current_index + 1} "
                        f"of {total_questions}"
                    )


                    st.progress(
                        current_index
                        / total_questions
                    )


                    st.write(
                        question_data["question"]
                    )


                    selected_answer = st.radio(
                        "Choose your answer:",
                        question_data["options"],
                        index=None,
                        key=f"quiz_question_{current_index}"
                    )


                    if not st.session_state.quiz_answered:

                        if st.button(
                            "Check Answer",
                            key="check_quiz_answer"
                        ):

                            if selected_answer is None:

                                st.warning(
                                    "Choose an answer first."
                                )

                            else:

                                st.session_state.quiz_answered = True


                                if (
                                    selected_answer
                                    == question_data["answer"]
                                ):

                                    st.session_state.quiz_score += 1


                                st.rerun()


                    else:

                        if (
                            selected_answer
                            == question_data["answer"]
                        ):

                            st.success(
                                "✅ Correct!"
                            )

                        else:

                            st.error(
                                "❌ Incorrect"
                            )

                            st.write(
                                "**Correct answer:** "
                                + question_data["answer"]
                            )


                        st.info(
                            "💡 "
                            + question_data["explanation"]
                        )


                        st.write(
                            f"Score: "
                            f"{st.session_state.quiz_score}"
                            f"/{current_index + 1}"
                        )


                        if st.button(
                            "Next Question ➡️",
                            use_container_width=True
                        ):

                            st.session_state.quiz_index += 1
                            st.session_state.quiz_answered = False

                            st.rerun()


        # ==================================
        # INTERACTIVE FLASHCARDS
        # ==================================

        with flashcard_tab:

            st.write(
                "Study important concepts "
                "one flashcard at a time."
            )


            if not st.session_state.flashcards:

                if st.button(
                    "🗂️ Generate Flashcards",
                    use_container_width=True
                ):

                    prompt = f"""
Create exactly 10 useful study flashcards
from the study material below.

Return ONLY valid JSON.

Use this structure:

[
    {{
        "question": "Question here",
        "answer": "Answer here"
    }}
]

Rules:

- Create exactly 10 flashcards.
- Keep the questions useful for studying.
- Keep answers clear and concise.
- Return JSON only.
- Do not use markdown.
- Do not use ```json.

STUDY MATERIAL:

{combined_text}
"""


                    with st.spinner(
                        "Creating flashcards..."
                    ):

                        response = client.responses.create(
                            model="gpt-5.6-luna",
                            input=prompt
                        )


                    try:

                        flashcard_data = json.loads(
                            response.output_text
                        )

                        st.session_state.flashcards = flashcard_data
                        st.session_state.flashcard_index = 0
                        st.session_state.show_flashcard_answer = False

                        st.rerun()


                    except json.JSONDecodeError:

                        st.error(
                            "The flashcards couldn't be "
                            "created correctly. Try again."
                        )


            else:

                total_cards = len(
                    st.session_state.flashcards
                )

                card_index = (
                    st.session_state.flashcard_index
                )

                current_card = (
                    st.session_state.flashcards[
                        card_index
                    ]
                )


                st.write(
                    f"### Card {card_index + 1} "
                    f"of {total_cards}"
                )


                st.progress(
                    (card_index + 1)
                    / total_cards
                )


                st.markdown("---")

                st.markdown(
                    "### ❓ Question"
                )

                st.write(
                    current_card["question"]
                )


                if not st.session_state.show_flashcard_answer:

                    if st.button(
                        "👀 Reveal Answer",
                        use_container_width=True
                    ):

                        st.session_state.show_flashcard_answer = True

                        st.rerun()


                else:

                    st.markdown(
                        "### ✅ Answer"
                    )

                    st.success(
                        current_card["answer"]
                    )


                st.markdown("---")


                left_col, right_col = st.columns(2)


                with left_col:

                    if st.button(
                        "⬅️ Previous",
                        disabled=(card_index == 0),
                        use_container_width=True
                    ):

                        st.session_state.flashcard_index -= 1
                        st.session_state.show_flashcard_answer = False

                        st.rerun()


                with right_col:

                    if st.button(
                        "Next ➡️",
                        disabled=(
                            card_index
                            == total_cards - 1
                        ),
                        use_container_width=True
                    ):

                        st.session_state.flashcard_index += 1
                        st.session_state.show_flashcard_answer = False

                        st.rerun()


                if st.button(
                    "🔄 Generate New Flashcards",
                    use_container_width=True
                ):

                    st.session_state.flashcards = []
                    st.session_state.flashcard_index = 0
                    st.session_state.show_flashcard_answer = False

                    st.rerun()


    else:

        st.info(
            "Upload a PDF to use document study mode."
        )