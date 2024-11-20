# Chat PDF: Conversational AI for PDF Files

Welcome to **Chat PDF**, a Streamlit-based application that enables interactive Q&A from PDF documents. This application uses modern language models and vector storage to provide precise answers directly from your uploaded PDFs.

## Features

- **Upload PDFs**: Process one or multiple PDF documents for Q&A.
- **Text Extraction**: Extracts text from PDFs using `PyPDF2`.
- **Chunk Splitting**: Splits extracted text into manageable chunks with overlap for better context understanding.
- **Vector Storage**: Leverages `FAISS` for efficient vector-based text search.
- **Language Model**: Utilizes `OpenAI GPT-3.5` for answering queries with contextual relevance.
- **Interactive Interface**: Powered by Streamlit for seamless user interaction.

---

## Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Required Python packages (listed in `requirements.txt`):
  ```text
  streamlit
  PyPDF2
  langchain
  langchain-community
  langchain-openai
  langchain-anthropic
  faiss-cpu
  spacy
  python-dotenv
  ```

---

## How It Works

1. **Upload PDFs**:  
   Upload one or multiple PDF files through the app interface.

2. **Process Text**:  
   The text is extracted, split into chunks, and stored as vectors in a FAISS database for efficient retrieval.

3. **Ask Questions**:  
   Enter your questions in the input box. The app retrieves relevant information from the processed PDFs and responds using an AI model.

4. **Accurate Contextual Responses**:  
   The system ensures that responses are grounded in the PDF content. If the answer is unavailable in the context, it explicitly states so.

---

## Key Components

### 1. **PDF Text Extraction**

Extracts raw text from uploaded PDF files using `PyPDF2`.

### 2. **Text Chunking**

Splits text into chunks of ~1000 characters with an overlap of 200 characters for better context retention.

### 3. **Vector Store**

Uses `FAISS` for storing text chunks as vectors, enabling efficient and scalable retrieval.

### 4. **Conversational AI**

Powered by `OpenAI GPT-3.5`, the AI generates detailed, context-aware answers to your questions.

### 5. **Streamlit Interface**

A user-friendly dashboard for uploading files, processing text, and interacting with the AI model.

---

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/your-repo/chat-pdf.git
   cd chat-pdf
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:  
   Create a `.env` file in the project root and add your API keys:

   ```text
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Interact with the app:
   - Upload PDFs.
   - Ask questions and get detailed, context-driven answers.

---

## File Structure

```plaintext
|-- app.py                   # Main application code
|-- requirements.txt         # Dependencies for the project
|-- .env                     # Environment variables (API keys)
|-- faiss_db/                # Local FAISS vector store (generated after processing)
```

---

## Limitations

- The app currently supports text-based PDFs only. Scanned images or non-text content are not processed.
- Model API keys (e.g., OpenAI) must be configured for successful operation.

---

## Future Improvements

- Add support for OCR to process scanned PDFs.
- Integrate additional AI models for enhanced flexibility.
- Optimize processing speed and scalability for larger documents.

---

## Contributions

Feel free to contribute by submitting issues or pull requests. For major changes, please open an issue first to discuss your ideas.

---

## License

This project is licensed under the MIT License.

---

Start exploring your PDFs with **Chat PDF**! 🚀
