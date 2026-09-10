import os
import math
import json
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv


# =========================================================
# ENVIRONMENT + API SETUP
# =========================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# This lets the same project work on Streamlit Cloud later
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None


st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .feature-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 12px;
    }

    .small-text {
        opacity: 0.7;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">📚 AI Study Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Learn with an AI tutor, study multiple PDFs,
    generate quizzes and flashcards, and track your progress.
    </div>
    """,
    unsafe_allow_html=True
)


if not api_key:
    st.error(
        "OpenAI API key was not found. "
        "Add OPENAI_API_KEY to your .env file "
        "or Streamlit secrets."
    )

    st.stop()


client = OpenAI(api_key=api_key)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {

    # General Tutor
    "general_messages": [],

    # PDF Chat
    "pdf_messages": [],

    # Documents
    "chunks": [],
    "embeddings": [],
    "current_files": None,

    # Quiz
    "quiz_questions": [],
    "quiz_index": 0,
    "quiz_score": 0,
    "quiz_answered": False,
    "quiz_recorded": False,

    # Flashcards
    "flashcards": [],
    "flashcard_index": 0,
    "show_flashcard_answer": False,

    # Generated study content
    "summary_text": "",

    # Progress
    "total_quizzes": 0,
    "total_questions": 0,
    "correct_answers": 0,
    "mistakes": [],

    # Extra mistake explanations
    "mistake_explanations": {}
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =========================================================
# SIDEBAR - STUDY SETTINGS
# =========================================================

with st.sidebar:

    st.title("⚙️ Study Settings")

    difficulty = st.selectbox(
        "Difficulty",
        [
            "Easy",
            "Medium",
            "Hard"
        ]
    )

    quiz_count = st.selectbox(
        "Quiz questions",
        [
            5,
            10,
            15
        ]
    )

    flashcard_count = st.selectbox(
        "Flashcards",
        [
            5,
            10,
            15,
            20
        ],
        index=1
    )

    explanation_mode = st.selectbox(
        "Explanation style",
        [
            "Normal",
            "Explain Simply",
            "Detailed Explanation",
            "Give Example",
            "Exam Answer"
        ]
    )

    st.divider()

    st.subheader("📊 Progress")

    total_questions = st.session_state.total_questions
    correct = st.session_state.correct_answers

    if total_questions > 0:

        accuracy = round(
            (correct / total_questions) * 100
        )

    else:

        accuracy = 0


    st.metric(
        "Quizzes Completed",
        st.session_state.total_quizzes
    )

    st.metric(
        "Questions Answered",
        total_questions
    )

    st.metric(
        "Accuracy",
        f"{accuracy}%"
    )

    st.metric(
        "Mistakes to Review",
        len(st.session_state.mistakes)
    )

    st.divider()

    st.caption(
        "Built with Python, Streamlit, "
        "OpenAI API, embeddings, and RAG."
    )


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def call_ai(prompt):

    try:

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )

        return response.output_text


    except Exception as error:

        st.error(
            f"AI request failed: {error}"
        )

        return None


# ---------------------------------------------------------
# SAFE JSON PARSER
# ---------------------------------------------------------

def parse_json_response(text):

    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):

        cleaned = cleaned.replace(
            "```json",
            ""
        )

        cleaned = cleaned.replace(
            "```",
            ""
        )

        cleaned = cleaned.strip()


    start = cleaned.find("[")
    end = cleaned.rfind("]")


    if start != -1 and end != -1:

        cleaned = cleaned[
            start:end + 1
        ]


    try:

        return json.loads(cleaned)

    except json.JSONDecodeError:

        return None


# =========================================================
# PDF CHUNKING
# =========================================================

def create_chunks(uploaded_files):

    chunks = []

    unreadable_files = []


    for uploaded_file in uploaded_files:

        try:

            uploaded_file.seek(0)

            reader = PdfReader(
                uploaded_file
            )

            readable_text_found = False


            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                try:

                    text = page.extract_text()

                except Exception:

                    text = None


                if not text:
                    continue


                text = text.strip()


                if not text:
                    continue


                readable_text_found = True


                chunk_size = 1200
                overlap = 200

                start = 0


                while start < len(text):

                    end = start + chunk_size

                    chunk_text = text[
                        start:end
                    ]


                    chunks.append({

                        "text": chunk_text,

                        "page": page_number,

                        "file": uploaded_file.name

                    })


                    start += (
                        chunk_size - overlap
                    )


            if not readable_text_found:

                unreadable_files.append(
                    uploaded_file.name
                )


        except Exception:

            unreadable_files.append(
                uploaded_file.name
            )


    return chunks, unreadable_files


# =========================================================
# EMBEDDINGS
# =========================================================

def create_embeddings(chunks):

    all_embeddings = []

    batch_size = 100


    for start in range(
        0,
        len(chunks),
        batch_size
    ):

        batch = chunks[
            start:start + batch_size
        ]


        texts = [
            chunk["text"]
            for chunk in batch
        ]


        try:

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )


            embeddings = [
                item.embedding
                for item in response.data
            ]


            all_embeddings.extend(
                embeddings
            )


        except Exception as error:

            st.error(
                f"Could not create document embeddings: {error}"
            )

            return []


    return all_embeddings


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(
    vector1,
    vector2
):

    dot_product = sum(
        a * b
        for a, b in zip(
            vector1,
            vector2
        )
    )


    magnitude1 = math.sqrt(
        sum(
            a * a
            for a in vector1
        )
    )


    magnitude2 = math.sqrt(
        sum(
            b * b
            for b in vector2
        )
    )


    if magnitude1 == 0:
        return 0


    if magnitude2 == 0:
        return 0


    return (
        dot_product
        /
        (magnitude1 * magnitude2)
    )


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_relevant_chunks(
    question,
    top_k=6
):

    if not st.session_state.embeddings:

        return []


    try:

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        )


        question_embedding = (
            response.data[0].embedding
        )


    except Exception as error:

        st.error(
            f"Could not search the documents: {error}"
        )

        return []


    scores = []


    for index, embedding in enumerate(
        st.session_state.embeddings
    ):

        score = cosine_similarity(
            question_embedding,
            embedding
        )


        scores.append(
            (
                score,
                index
            )
        )


    scores.sort(
        reverse=True
    )


    best_chunks = []


    for score, index in scores[
        :top_k
    ]:

        best_chunks.append(
            st.session_state.chunks[
                index
            ]
        )


    return best_chunks


# =========================================================
# EXPLANATION MODE
# =========================================================

def explanation_instruction(mode):

    if mode == "Explain Simply":

        return """
Explain the answer using very simple,
beginner-friendly language.
Avoid unnecessary technical jargon.
"""


    if mode == "Detailed Explanation":

        return """
Give a detailed explanation.
Explain the reasoning and important concepts
step-by-step.
"""


    if mode == "Give Example":

        return """
Explain the answer and include at least
one useful example.
"""


    if mode == "Exam Answer":

        return """
Answer like a strong college exam response.
Be concise, accurate, organized, and include
the key points a professor would expect.
"""


    return """
Explain clearly at a college-student level.
"""


# =========================================================
# DOCUMENT Q&A WITH INLINE CITATIONS
# =========================================================

def ask_pdf(
    question,
    mode
):

    relevant_chunks = (
        retrieve_relevant_chunks(
            question
        )
    )


    if not relevant_chunks:

        return (
            "I couldn't retrieve information "
            "from the uploaded documents.",
            []
        )


    context = ""

    source_labels = []


    for chunk in relevant_chunks:

        citation = (
            f"[{chunk['file']}, "
            f"p. {chunk['page']}]"
        )


        source_labels.append(
            citation
        )


        context += f"""
SOURCE: {citation}

{chunk["text"]}

"""


    instruction = (
        explanation_instruction(mode)
    )


    prompt = f"""
You are an AI Study Assistant.

Answer the student's question using ONLY
the provided study material.

{instruction}

IMPORTANT CITATION RULES:

Every important statement based on the
documents should include the relevant citation.

Use citations exactly like:

[Lecture1.pdf, p. 4]

or

[Chapter2.pdf, p. 17]

Do not invent page numbers.

If information comes from multiple sources,
you may cite multiple sources.

If the answer is not supported by the provided
material, clearly say:

"I could not find enough information in the
uploaded documents."

STUDY MATERIAL:

{context}

STUDENT QUESTION:

{question}
"""


    answer = call_ai(
        prompt
    )


    if not answer:

        answer = (
            "The AI could not generate "
            "an answer."
        )


    return (
        answer,
        list(dict.fromkeys(
            source_labels
        ))
    )


# =========================================================
# MATERIAL FOR SUMMARY / QUIZ / FLASHCARDS
# =========================================================

def build_study_material(
    max_characters=60000
):

    if not st.session_state.chunks:

        return ""


    material = ""

    used = 0


    for chunk in st.session_state.chunks:

        section = (
            f"\nSOURCE: "
            f"[{chunk['file']}, "
            f"p. {chunk['page']}]\n"
            f"{chunk['text']}\n"
        )


        if (
            used + len(section)
            > max_characters
        ):

            break


        material += section

        used += len(section)


    return material


# =========================================================
# DOWNLOAD BUILDERS
# =========================================================

def build_quiz_download():

    text = "AI Study Assistant - Quiz\n\n"


    for index, question in enumerate(
        st.session_state.quiz_questions,
        start=1
    ):

        text += (
            f"Question {index}\n"
            f"{question.get('question', '')}\n\n"
        )


        for option in question.get(
            "options",
            []
        ):

            text += (
                f"- {option}\n"
            )


        text += (
            "\nCorrect Answer: "
            f"{question.get('answer', '')}\n"
        )


        text += (
            "Explanation: "
            f"{question.get('explanation', '')}\n"
        )


        text += "\n--------------------\n\n"


    return text


def build_flashcard_download():

    text = (
        "AI Study Assistant - Flashcards\n\n"
    )


    for index, card in enumerate(
        st.session_state.flashcards,
        start=1
    ):

        text += (
            f"Flashcard {index}\n"
        )


        text += (
            f"Question: "
            f"{card.get('question', '')}\n"
        )


        text += (
            f"Answer: "
            f"{card.get('answer', '')}\n\n"
        )


    return text


# =========================================================
# MAIN NAVIGATION
# =========================================================

tutor_tab, pdf_tab, progress_tab, about_tab = (
    st.tabs(
        [
            "💬 AI Tutor",
            "📚 Study Documents",
            "📊 Progress",
            "ℹ️ About"
        ]
    )
)


# =========================================================
# AI TUTOR
# =========================================================

with tutor_tab:

    st.header(
        "💬 AI Tutor"
    )

    st.write(
        "Ask questions without uploading "
        "any documents."
    )


    col1, col2 = st.columns(
        [4, 1]
    )


    with col2:

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
            key="clear_general"
        ):

            st.session_state.general_messages = []

            st.rerun()


    for message in (
        st.session_state.general_messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    tutor_question = st.chat_input(
        "Ask your AI Tutor anything...",
        key="general_chat"
    )


    if tutor_question:

        st.session_state.general_messages.append(
            {
                "role": "user",
                "content": tutor_question
            }
        )


        with st.chat_message("user"):

            st.markdown(
                tutor_question
            )


        conversation = ""


        for message in (
            st.session_state.general_messages[
                -10:
            ]
        ):

            conversation += (
                f"{message['role']}: "
                f"{message['content']}\n\n"
            )


        instruction = (
            explanation_instruction(
                explanation_mode
            )
        )


        prompt = f"""
You are an AI Tutor for a college student.

{instruction}

For programming questions:

- explain code clearly
- use beginner-friendly examples
- explain important lines when useful
- do not unnecessarily overcomplicate the answer

Conversation:

{conversation}

Answer the student's latest message.
"""


        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                answer = call_ai(
                    prompt
                )


            if answer:

                st.markdown(
                    answer
                )


                st.session_state.general_messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


# =========================================================
# STUDY DOCUMENTS
# =========================================================

with pdf_tab:

    st.header(
        "📚 Study Your Documents"
    )

    st.write(
        "Upload lecture slides, textbook chapters, "
        "study guides, or notes together."
    )


    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF files."
    )


    # -----------------------------------------------------
    # PROCESS FILES
    # -----------------------------------------------------

    if uploaded_files:

        file_signature = tuple(
            (
                file.name,
                file.size
            )
            for file in uploaded_files
        )


        if (
            st.session_state.current_files
            != file_signature
        ):

            with st.spinner(
                "Reading and indexing your documents..."
            ):

                chunks, unreadable = (
                    create_chunks(
                        uploaded_files
                    )
                )


                if not chunks:

                    st.error(
                        "No readable text was found. "
                        "The PDFs may be scanned images "
                        "or corrupted."
                    )

                    st.stop()


                embeddings = (
                    create_embeddings(
                        chunks
                    )
                )


                if not embeddings:

                    st.stop()


                st.session_state.chunks = (
                    chunks
                )

                st.session_state.embeddings = (
                    embeddings
                )

                st.session_state.current_files = (
                    file_signature
                )


                # Reset document-specific content
                st.session_state.pdf_messages = []

                st.session_state.quiz_questions = []

                st.session_state.quiz_index = 0

                st.session_state.quiz_score = 0

                st.session_state.quiz_answered = False

                st.session_state.quiz_recorded = False

                st.session_state.flashcards = []

                st.session_state.flashcard_index = 0

                st.session_state.show_flashcard_answer = False

                st.session_state.summary_text = ""


            st.success(
                "Documents indexed successfully!"
            )


            if unreadable:

                st.warning(
                    "Some PDFs contained little or no "
                    "extractable text: "
                    + ", ".join(unreadable)
                )


        # -------------------------------------------------
        # DOCUMENT INFORMATION
        # -------------------------------------------------

        st.subheader(
            "📄 Loaded Documents"
        )


        for file in uploaded_files:

            st.write(
                f"• {file.name}"
            )


        st.caption(
            f"{len(uploaded_files)} document(s) | "
            f"{len(st.session_state.chunks)} searchable sections"
        )


        # -------------------------------------------------
        # PDF CHAT
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "🔎 Ask Your Documents"
        )


        clear_col, info_col = st.columns(
            [1, 4]
        )


        with clear_col:

            if st.button(
                "🗑️ Clear Chat",
                key="clear_document_chat",
                use_container_width=True
            ):

                st.session_state.pdf_messages = []

                st.rerun()


        for message in (
            st.session_state.pdf_messages
        ):

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )


                if message.get("sources"):

                    with st.expander(
                        "📍 Retrieved Sources"
                    ):

                        for source in (
                            message["sources"]
                        ):

                            st.write(
                                source
                            )


        pdf_question = st.chat_input(
            "Ask something about your documents...",
            key="pdf_chat"
        )


        if pdf_question:

            st.session_state.pdf_messages.append(
                {
                    "role": "user",
                    "content": pdf_question
                }
            )


            with st.chat_message(
                "user"
            ):

                st.markdown(
                    pdf_question
                )


            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Searching your documents..."
                ):

                    answer, sources = (
                        ask_pdf(
                            pdf_question,
                            explanation_mode
                        )
                    )


                st.markdown(
                    answer
                )


                if sources:

                    with st.expander(
                        "📍 Retrieved Sources"
                    ):

                        for source in sources:

                            st.write(
                                source
                            )


            st.session_state.pdf_messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )


        # =================================================
        # STUDY TOOLS
        # =================================================

        st.divider()

        st.header(
            "🧠 Study Tools"
        )


        summary_tab, quiz_tab, flashcards_tab, mistakes_tab = (
            st.tabs(
                [
                    "📝 Summary",
                    "❓ Quiz",
                    "🗂️ Flashcards",
                    "❌ Review Mistakes"
                ]
            )
        )


        study_material = (
            build_study_material()
        )


        # =================================================
        # SUMMARY
        # =================================================

        with summary_tab:

            st.subheader(
                "📝 AI Study Summary"
            )


            st.write(
                "Create an exam-ready summary "
                "from your uploaded material."
            )


            if st.button(
                "✨ Generate Summary",
                use_container_width=True
            ):

                prompt = f"""
You are an AI Study Assistant.

Create a {difficulty.lower()} level study
summary from the material below.

Organize the summary with:

- Main ideas
- Important concepts
- Important definitions
- Important facts
- Things likely worth remembering for an exam
- A short final review section

When useful, include the source citation exactly
as it appears in the material.

STUDY MATERIAL:

{study_material}
"""


                with st.spinner(
                    "Creating your summary..."
                ):

                    result = call_ai(
                        prompt
                    )


                if result:

                    st.session_state.summary_text = (
                        result
                    )


            if (
                st.session_state.summary_text
            ):

                st.markdown(
                    st.session_state.summary_text
                )


                st.download_button(
                    "⬇️ Download Summary",
                    data=st.session_state.summary_text,
                    file_name="study_summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )


        # =================================================
        # QUIZ
        # =================================================

        with quiz_tab:

            st.subheader(
                "❓ Interactive Quiz"
            )


            st.caption(
                f"Difficulty: {difficulty} | "
                f"Questions: {quiz_count}"
            )


            # ---------------------------------------------
            # CREATE QUIZ
            # ---------------------------------------------

            if (
                not st.session_state.quiz_questions
            ):

                if st.button(
                    "🎯 Generate Quiz",
                    use_container_width=True
                ):

                    prompt = f"""
Create exactly {quiz_count}
multiple-choice questions using ONLY
the study material below.

Difficulty:
{difficulty}

Return ONLY valid JSON.

Use this structure:

[
    {{
        "question": "Question text",
        "options": [
            "Answer option 1",
            "Answer option 2",
            "Answer option 3",
            "Answer option 4"
        ],
        "answer": "Exact correct option",
        "explanation": "Short explanation of why the answer is correct",
        "topic": "Main concept being tested"
    }}
]

RULES:

- Exactly {quiz_count} questions
- Exactly 4 options per question
- Only one correct answer
- "answer" must exactly match an option
- Questions must come from the study material
- Match the requested difficulty
- Return JSON only
- No markdown
- No code fences

STUDY MATERIAL:

{study_material}
"""


                    with st.spinner(
                        "Building your quiz..."
                    ):

                        result = call_ai(
                            prompt
                        )


                    quiz_data = (
                        parse_json_response(
                            result
                        )
                    )


                    if (
                        isinstance(
                            quiz_data,
                            list
                        )
                        and quiz_data
                    ):

                        st.session_state.quiz_questions = (
                            quiz_data
                        )

                        st.session_state.quiz_index = 0

                        st.session_state.quiz_score = 0

                        st.session_state.quiz_answered = False

                        st.session_state.quiz_recorded = False

                        st.rerun()


                    else:

                        st.error(
                            "The quiz could not be formatted "
                            "correctly. Click Generate Quiz "
                            "and try again."
                        )


            # ---------------------------------------------
            # QUIZ EXISTS
            # ---------------------------------------------

            else:

                questions = (
                    st.session_state.quiz_questions
                )

                index = (
                    st.session_state.quiz_index
                )

                total = len(
                    questions
                )


                # =========================================
                # QUIZ FINISHED
                # =========================================

                if index >= total:

                    score = (
                        st.session_state.quiz_score
                    )


                    percentage = round(
                        (
                            score
                            / total
                        )
                        * 100
                    )


                    # Record quiz only once
                    if (
                        not st.session_state.quiz_recorded
                    ):

                        st.session_state.total_quizzes += 1

                        st.session_state.quiz_recorded = True


                    st.success(
                        "🎉 Quiz Complete!"
                    )


                    score_col, percent_col = (
                        st.columns(2)
                    )


                    with score_col:

                        st.metric(
                            "Score",
                            f"{score}/{total}"
                        )


                    with percent_col:

                        st.metric(
                            "Accuracy",
                            f"{percentage}%"
                        )


                    st.progress(
                        score / total
                    )


                    if percentage >= 90:

                        st.success(
                            "🏆 Excellent work!"
                        )

                    elif percentage >= 75:

                        st.info(
                            "👏 Good job! Review the "
                            "questions you missed."
                        )

                    elif percentage >= 60:

                        st.warning(
                            "📖 You're getting there. "
                            "Review the weaker topics."
                        )

                    else:

                        st.warning(
                            "💪 Review the material "
                            "and try again."
                        )


                    st.download_button(
                        "⬇️ Download Quiz + Answers",
                        data=build_quiz_download(),
                        file_name="study_quiz.txt",
                        mime="text/plain",
                        use_container_width=True
                    )


                    if st.button(
                        "🔄 Generate Another Quiz",
                        use_container_width=True
                    ):

                        st.session_state.quiz_questions = []

                        st.session_state.quiz_index = 0

                        st.session_state.quiz_score = 0

                        st.session_state.quiz_answered = False

                        st.session_state.quiz_recorded = False

                        st.rerun()


                # =========================================
                # CURRENT QUESTION
                # =========================================

                else:

                    question = questions[
                        index
                    ]


                    st.write(
                        f"### Question {index + 1} "
                        f"of {total}"
                    )


                    st.progress(
                        index / total
                    )


                    if question.get(
                        "topic"
                    ):

                        st.caption(
                            f"Topic: "
                            f"{question['topic']}"
                        )


                    st.markdown(
                        f"### {question['question']}"
                    )


                    options = (
                        question.get(
                            "options",
                            []
                        )
                    )


                    selected = st.radio(
                        "Choose your answer:",
                        options,
                        index=None,
                        key=f"quiz_answer_{index}"
                    )


                    # -------------------------------------
                    # CHECK ANSWER
                    # -------------------------------------

                    if (
                        not st.session_state.quiz_answered
                    ):

                        if st.button(
                            "✅ Check Answer",
                            use_container_width=True
                        ):

                            if selected is None:

                                st.warning(
                                    "Choose an answer first."
                                )


                            else:

                                correct_answer = (
                                    question.get(
                                        "answer"
                                    )
                                )


                                st.session_state.total_questions += 1


                                if (
                                    selected
                                    == correct_answer
                                ):

                                    st.session_state.quiz_score += 1

                                    st.session_state.correct_answers += 1


                                else:

                                    st.session_state.mistakes.append(
                                        {
                                            "question": question.get(
                                                "question",
                                                ""
                                            ),

                                            "your_answer": selected,

                                            "correct_answer": correct_answer,

                                            "explanation": question.get(
                                                "explanation",
                                                ""
                                            ),

                                            "topic": question.get(
                                                "topic",
                                                "Unknown topic"
                                            )
                                        }
                                    )


                                st.session_state.quiz_answered = True

                                st.rerun()


                    # -------------------------------------
                    # FEEDBACK
                    # -------------------------------------

                    else:

                        correct_answer = (
                            question.get(
                                "answer"
                            )
                        )


                        if (
                            selected
                            == correct_answer
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
                                f"{correct_answer}"
                            )


                        explanation = (
                            question.get(
                                "explanation"
                            )
                        )


                        if explanation:

                            st.info(
                                "💡 "
                                + explanation
                            )


                        st.write(
                            f"Current score: "
                            f"{st.session_state.quiz_score}/"
                            f"{index + 1}"
                        )


                        if st.button(
                            "Next Question ➡️",
                            use_container_width=True
                        ):

                            st.session_state.quiz_index += 1

                            st.session_state.quiz_answered = False

                            st.rerun()


        # =================================================
        # FLASHCARDS
        # =================================================

        with flashcards_tab:

            st.subheader(
                "🗂️ Interactive Flashcards"
            )


            st.caption(
                f"Difficulty: {difficulty} | "
                f"Cards: {flashcard_count}"
            )


            if (
                not st.session_state.flashcards
            ):

                if st.button(
                    "✨ Generate Flashcards",
                    use_container_width=True
                ):

                    prompt = f"""
Create exactly {flashcard_count}
study flashcards from the material below.

Difficulty:
{difficulty}

Return ONLY valid JSON.

Format:

[
    {{
        "question": "Question here",
        "answer": "Answer here"
    }}
]

Rules:

- Exactly {flashcard_count} flashcards
- Important concepts only
- Questions should help with studying
- Answers should be clear
- Match the requested difficulty
- Return JSON only
- Do not use markdown
- Do not use code fences

STUDY MATERIAL:

{study_material}
"""


                    with st.spinner(
                        "Creating flashcards..."
                    ):

                        result = call_ai(
                            prompt
                        )


                    cards = (
                        parse_json_response(
                            result
                        )
                    )


                    if (
                        isinstance(
                            cards,
                            list
                        )
                        and cards
                    ):

                        st.session_state.flashcards = (
                            cards
                        )

                        st.session_state.flashcard_index = 0

                        st.session_state.show_flashcard_answer = False

                        st.rerun()


                    else:

                        st.error(
                            "The flashcards could not be "
                            "formatted correctly. Try again."
                        )


            else:

                cards = (
                    st.session_state.flashcards
                )

                card_index = (
                    st.session_state.flashcard_index
                )

                card = cards[
                    card_index
                ]


                st.write(
                    f"### Card {card_index + 1} "
                    f"of {len(cards)}"
                )


                st.progress(
                    (
                        card_index + 1
                    )
                    / len(cards)
                )


                st.markdown(
                    "---"
                )


                st.markdown(
                    "### ❓ Question"
                )


                st.markdown(
                    card.get(
                        "question",
                        ""
                    )
                )


                if (
                    not
                    st.session_state.show_flashcard_answer
                ):

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
                        card.get(
                            "answer",
                            ""
                        )
                    )


                st.markdown(
                    "---"
                )


                previous_col, next_col = (
                    st.columns(2)
                )


                with previous_col:

                    if st.button(
                        "⬅️ Previous",
                        disabled=(
                            card_index == 0
                        ),
                        use_container_width=True
                    ):

                        st.session_state.flashcard_index -= 1

                        st.session_state.show_flashcard_answer = False

                        st.rerun()


                with next_col:

                    if st.button(
                        "Next ➡️",
                        disabled=(
                            card_index
                            == len(cards) - 1
                        ),
                        use_container_width=True
                    ):

                        st.session_state.flashcard_index += 1

                        st.session_state.show_flashcard_answer = False

                        st.rerun()


                st.download_button(
                    "⬇️ Download Flashcards",
                    data=build_flashcard_download(),
                    file_name="study_flashcards.txt",
                    mime="text/plain",
                    use_container_width=True
                )


                if st.button(
                    "🔄 Generate New Flashcards",
                    use_container_width=True
                ):

                    st.session_state.flashcards = []

                    st.session_state.flashcard_index = 0

                    st.session_state.show_flashcard_answer = False

                    st.rerun()


        # =================================================
        # REVIEW MISTAKES
        # =================================================

        with mistakes_tab:

            st.subheader(
                "❌ Review Your Mistakes"
            )


            if (
                not st.session_state.mistakes
            ):

                st.success(
                    "No mistakes to review yet. 🎉"
                )


            else:

                st.write(
                    "These are questions you answered "
                    "incorrectly."
                )


                for index, mistake in enumerate(
                    reversed(
                        st.session_state.mistakes
                    )
                ):

                    real_index = (
                        len(
                            st.session_state.mistakes
                        )
                        - 1
                        - index
                    )


                    with st.expander(
                        f"❌ "
                        f"{mistake.get('topic', 'Review Topic')}"
                    ):

                        st.write(
                            "**Question:**"
                        )

                        st.write(
                            mistake.get(
                                "question",
                                ""
                            )
                        )


                        st.write(
                            "**Your answer:** "
                            f"{mistake.get('your_answer', '')}"
                        )


                        st.write(
                            "**Correct answer:** "
                            f"{mistake.get('correct_answer', '')}"
                        )


                        st.info(
                            mistake.get(
                                "explanation",
                                ""
                            )
                        )


                        if st.button(
                            "🧠 Explain This Concept",
                            key=f"explain_mistake_{real_index}"
                        ):

                            prompt = f"""
You are an AI Tutor.

A student got this question wrong.

TOPIC:
{mistake.get("topic", "")}

QUESTION:
{mistake.get("question", "")}

STUDENT ANSWER:
{mistake.get("your_answer", "")}

CORRECT ANSWER:
{mistake.get("correct_answer", "")}

Explain why the correct answer is right
and teach the underlying concept in
beginner-friendly language.

Include a simple example if useful.
"""


                            with st.spinner(
                                "Explaining..."
                            ):

                                explanation = (
                                    call_ai(
                                        prompt
                                    )
                                )


                            if explanation:

                                st.session_state.mistake_explanations[
                                    real_index
                                ] = explanation


                        if (
                            real_index
                            in st.session_state.mistake_explanations
                        ):

                            st.markdown(
                                st.session_state.mistake_explanations[
                                    real_index
                                ]
                            )


                if st.button(
                    "🗑️ Clear Mistake History",
                    use_container_width=True
                ):

                    st.session_state.mistakes = []

                    st.session_state.mistake_explanations = {}

                    st.rerun()


    else:

        st.info(
            "👆 Upload one or more PDFs to begin studying."
        )


        st.markdown(
            """
            ### What you can upload

            Lecture slides, class notes, textbook chapters,
            study guides, assignments, or other PDF material
            can be studied together.
            """
        )


# =========================================================
# PROGRESS DASHBOARD
# =========================================================

with progress_tab:

    st.header(
        "📊 Study Progress"
    )


    total = (
        st.session_state.total_questions
    )

    correct = (
        st.session_state.correct_answers
    )

    incorrect = (
        total - correct
    )


    if total > 0:

        accuracy = round(
            (correct / total) * 100
        )

    else:

        accuracy = 0


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            "Quizzes",
            st.session_state.total_quizzes
        )


    with col2:

        st.metric(
            "Questions",
            total
        )


    with col3:

        st.metric(
            "Correct",
            correct
        )


    with col4:

        st.metric(
            "Accuracy",
            f"{accuracy}%"
        )


    st.divider()


    if total > 0:

        st.subheader(
            "Overall Accuracy"
        )

        st.progress(
            correct / total
        )


        st.write(
            f"✅ Correct: {correct}"
        )

        st.write(
            f"❌ Incorrect: {incorrect}"
        )


    else:

        st.info(
            "Complete a quiz to start tracking "
            "your progress."
        )


    if (
        st.session_state.mistakes
    ):

        st.divider()

        st.subheader(
            "Topics Needing Review"
        )


        topic_counts = {}


        for mistake in (
            st.session_state.mistakes
        ):

            topic = mistake.get(
                "topic",
                "Unknown Topic"
            )


            topic_counts[topic] = (
                topic_counts.get(
                    topic,
                    0
                )
                + 1
            )


        sorted_topics = sorted(
            topic_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )


        for topic, count in (
            sorted_topics
        ):

            st.write(
                f"🔸 {topic} — "
                f"{count} mistake(s)"
            )


# =========================================================
# ABOUT PAGE
# =========================================================

with about_tab:

    st.header(
        "ℹ️ About AI Study Assistant"
    )


    st.write(
        """
        AI Study Assistant is an AI-powered learning
        application built to help students understand
        course material more efficiently.
        """
    )


    st.subheader(
        "✨ Features"
    )


    st.markdown(
        """
        - 💬 General AI tutoring
        - 📚 Multiple PDF study
        - 🔎 Retrieval-Augmented Generation (RAG)
        - 📍 PDF filename and page citations
        - 📝 AI summaries
        - ❓ Interactive quizzes
        - 🗂️ Interactive flashcards
        - 🎚️ Difficulty settings
        - 🧠 Multiple explanation styles
        - 📊 Progress tracking
        - ❌ Mistake review
        - 💾 Downloadable study material
        """
    )


    st.subheader(
        "🛠️ Technology"
    )


    st.write(
        """
        Python • Streamlit • OpenAI API •
        Embeddings • RAG • PyPDF •
        Session State
        """
    )


    st.subheader(
        "🛡️ Privacy"
    )


    st.write(
        """
        API keys are stored outside the source code
        using environment variables or deployment
        secrets.
        """
    )