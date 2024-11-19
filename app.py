import streamlit as st
from PyPDF2 import PdfReader
from pptx import Presentation
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

load_dotenv()

# Configure Tesseract executable path (if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # For Windows

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_image_text(images):
    text = ""
    for image in images:
        img = Image.open(image)
        text += pytesseract.image_to_string(img)
    return text

def get_image_based_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        images = convert_from_path(pdf)
        for img in images:
            text += pytesseract.image_to_string(img)
    return text

def get_ppt_text(ppt_docs):
    text = ""
    for ppt in ppt_docs:
        presentation = Presentation(ppt)
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + " "
    return text

def get_doc_text(doc_docs):
    text = ""
    for doc in doc_docs:
        doc_file = docx.Document(doc)
        for para in doc_file.paragraphs:
            text += para.text + " "
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is given in points make sure to give the points in different lines, 
    if the answer is not in the provided context just say, "Answer cannot be found", don't provide the wrong answer.
    Context: \n{context}\n
    Question: \n{question}\n

    Answer:  
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    # Return a default message if the answer is empty
    answer = response.get("output_text", "").strip()
    if not answer:
        return "Answer cannot be found"
    return answer

def main():
    st.set_page_config(page_title="ChatDoc with OCR", page_icon=":books:")
    st.markdown("""
    <style>
    .main { padding: 20px; }
    .header { font-size: 24px; font-weight: bold; padding: 20px 0; }
    </style>
    """, unsafe_allow_html=True)

    st.header("ChatDoc with OCR :books:")
    st.markdown('<div class="header">Chat with your Documents and Images Here</div>', unsafe_allow_html=True)

    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = []

    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.subheader("Your Uploaded Files:")
        doc_files = st.file_uploader("Upload PDFs, PPTs, DOCs, or Images:", accept_multiple_files=True, type=['pdf', 'pptx', 'docx', 'png', 'jpg', 'jpeg'])
        if st.button('Process'):
            if doc_files:
                with st.spinner('Processing...'):
                    raw_text = ""
                    pdf_files = [file for file in doc_files if file.name.endswith('.pdf')]
                    ppt_files = [file for file in doc_files if file.name.endswith('.pptx')]
                    docx_files = [file for file in doc_files if file.name.endswith('.docx')]
                    img_files = [file for file in doc_files if file.name.endswith(('.png', '.jpg', '.jpeg'))]

                    if pdf_files:
                        raw_text += get_pdf_text(pdf_files)
                        raw_text += get_image_based_pdf_text(pdf_files)
                    if ppt_files:
                        raw_text += get_ppt_text(ppt_files)
                    if docx_files:
                        raw_text += get_doc_text(docx_files)
                    if img_files:
                        raw_text += get_image_text(img_files)

                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.processed = True
                    st.success("Processing complete. You can now ask questions.")
            else:
                st.warning("Please upload at least one document or image before processing.")

    if st.session_state.processed:
        user_question = st.text_input("Ask a question:", "", key="question", help="Type your question here")
        if user_question:
            with st.spinner('Fetching answer...'):
                answer = user_input(user_question)
                if not st.session_state.qa_pairs or st.session_state.qa_pairs[-1][0] != user_question:
                    st.session_state.qa_pairs.append((user_question, answer))

        for question, answer in reversed(st.session_state.qa_pairs):
            st.markdown(f"<strong>Question:</strong> {question}")
            st.markdown(f"<strong>Answer:</strong> {answer}")
    else:
        st.info("Please upload and process a document to start asking questions.")

if __name__ == '__main__':
    main()
