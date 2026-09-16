# AI Study Assistant

An AI-powered study application that lets users upload PDF documents and ask questions about the material - built with a Retrieval-Augmented Generation (RAG) pipeline so answers are grounded in the actual document content, not just the model's general knowledge.

## Features

- **PDF upload & parsing** — upload any PDF and the app extracts and processes the text
- **Question answering** — ask natural-language questions about the uploaded material and get answers sourced from the document
- **Page-referenced answers** — responses link back to the specific page they were drawn from, so you can verify the source
- **Document summaries** — auto-generated summaries of uploaded content
- **Interactive quizzes** — generated quiz questions based on the document
- **Flashcards** — auto-generated flashcards for studying key concepts

## How It Works (RAG Pipeline)

1. **Chunking** — uploaded PDFs are split into smaller text chunks using PyPDF, sized to balance context and retrieval precision
2. **Embeddings** — each chunk is converted into a vector embedding via the OpenAI API
3. **Retrieval** — when a user asks a question, the question is embedded and compared against document chunk embeddings using cosine similarity to find the most relevant sections
4. **Generation** — the most relevant chunks are passed to the OpenAI API as context, which generates an answer grounded in that retrieved content


## Tech Stack

- **Python** — core application logic
- **Streamlit** — user interface
- **OpenAI API** — embeddings and answer generation
- **PyPDF** — PDF text extraction
- **Session state** (Streamlit) — manages conversation history, document data, quiz state, and flashcard state across user interactions

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd ai-study-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

4. Run the app:
   ```bash
   streamlit run main.py
   ```

5. Open the local URL Streamlit provides, upload a PDF, and start asking questions.

## Project Structure

```
ai-study-assistant/
├── .gitignore              
├── main.py               # Main Application Chunking, embedding, and retrieval logic
├── requirements.txt      # Python dependencies
└── README.md
```


## Future Improvements

- Support for multiple document formats (e.g., .docx, .txt)
- Persistent storage for past sessions/documents
- Improved chunking strategy for longer documents

## Author

Abdullah Abdul Rahman
[LinkedIn](https://linkedin.com/in/abdullahar16) · [GitHub](https://github.com/abdullahrahmanabdul79-cyber)
