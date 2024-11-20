# Chat PDF: Conversational AI for PDF Files

Welcome to **Chat PDF**, an innovative application that turns static PDF content into dynamic, interactive conversations. With advanced AI and retrieval techniques, Chat PDF enables you to extract insights and answers directly from your documents.

---

## 📌 Key Features

### 🚀 **Effortless PDF Interaction**

- Upload one or more PDF files and ask questions directly.
- Get accurate, context-driven answers within seconds.

### 📚 **Advanced Text Processing**

- Extracts text from PDFs using **PyPDF2**.
- Splits content into manageable chunks for better retrieval accuracy.

### 🔍 **Efficient Search with FAISS**

- Utilizes FAISS (Facebook AI Similarity Search) for fast, scalable text search.

### 🤖 **Powerful Conversational AI**

- Leverages **OpenAI GPT-3.5** for context-aware and detailed responses.

### 🖥️ **Interactive Interface**

- Built on **Streamlit**, ensuring a user-friendly and seamless experience.

---

## 👨‍💻 Authors

This project was developed by:

- [**Anurag**](https://github.com/the-wilful-youth)
- [**Dhawal**](https://github.com/techbolt)
- [**Animesh**]

Explore their GitHub profiles to discover more of their amazing work!

---

## ⚙️ Prerequisites

To run this application, make sure you have the following:

- **Python**: Version 3.8 or later
- **Dependencies**: Install the required Python packages listed in `requirements.txt`:
  ```bash
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

## 🛠️ How It Works

### 1. **Upload PDFs**

Upload one or multiple PDF files through the app interface.

### 2. **Process Text**

- Extracts raw text from PDFs.
- Splits the text into chunks (~1000 characters each) with overlaps for context retention.
- Stores chunks as vectors in a FAISS database for efficient search.

### 3. **Ask Questions**

Submit your question in the app. The system retrieves relevant information and generates a response using the AI model.

### 4. **Accurate Responses**

If the answer is not found in the uploaded PDFs, the app will let you know, avoiding incorrect responses.

---

## 🏗️ Getting Started

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-repo/chat-pdf.git
   cd chat-pdf
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Keys**:  
   Create a `.env` file and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the Application**:

   ```bash
   streamlit run main.py
   ```

5. **Interact with the App**:
   - Upload your PDF documents.
   - Ask questions and receive answers with ease.

---

## 📂 File Structure

```plaintext
|-- main.py                   # Main application script
|-- requirements.txt         # List of dependencies
|-- .env                     # Environment variables for API keys
|-- faiss_db/                # Local FAISS vector store (auto-generated)
```

---

## 🚧 Limitations

- **Text-Based PDFs Only**: Currently, only text-based PDFs are supported. PDFs with scanned images require OCR (not yet implemented).
- **API Dependency**: Requires a valid OpenAI API key for operation.

---

## 🌟 Future Enhancements

- **OCR Integration**: Add support for image-based and scanned PDFs.
- **Additional AI Models**: Expand compatibility with more language models.
- **Performance Optimization**: Improve scalability for large document sets.

---

## 🤝 Contributions

We welcome your ideas and contributions!

- Submit issues or pull requests to help improve the project.
- For major changes, please open an issue first to discuss your proposal.

---

## 📜 License

This project is licensed under the **MIT License**.

---

### Start your journey of exploring PDFs like never before with **Chat PDF**! 🚀
