import streamlit as st
from PyPDF2 import PdfReader
from pptx import Presentation
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import pytesseract
import sympy as sp

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text_images(pdf_docs):
    text = ""
    images = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text += page.extract_text()
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                images.append((page_num, image))
    return text, images

def get_ppt_text_images(ppt_docs):
    text = ""
    images = []
    for ppt in ppt_docs:
        presentation = Presentation(ppt)
        for slide_num, slide in enumerate(presentation.slides):
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + " "
                if shape.shape_type == 13:  # Shape type 13 is Picture
                    image = shape.image
                    if image:
                        img_bytes = io.BytesIO(image.blob)
                        images.append((slide_num, Image.open(img_bytes)))
    return text, images

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
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is given in points make sure to give the points in different lines, if the answer is not in the
    provided context just say, "Answer cannot be found", don't provide the wrong answer.
    Context: \n{context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, images):
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
        answer = "Answer cannot be found"

    # If user question is image-related, add relevant images
    if "image" in user_question.lower():
        for page_num, img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            answer += f'\n\nPage {page_num + 1}:\n![Image](data:image/png;base64,{img_str})'

    return answer

def main():
    st.set_page_config(page_title="ChatDoc", page_icon=":books:")
    st.header("ChatDoc :books:")
    st.markdown('<div class="header">Chat with your Documents Here</div>', unsafe_allow_html=True)

    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = []

    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.subheader("Your Uploaded Files : ")
        doc_files = st.file_uploader("Upload your PDFs, PPTs, or DOCs here:", accept_multiple_files=True, type=['pdf', 'pptx', 'docx'])
        if st.button('Process'):
            if doc_files:
                with st.spinner('Processing...'):
                    raw_text = ""
                    images = []
                    pdf_files = [file for file in doc_files if file.name.endswith('.pdf')]
                    ppt_files = [file for file in doc_files if file.name.endswith('.pptx')]
                    docx_files = [file for file in doc_files if file.name.endswith('.docx')]
                    
                    if pdf_files:
                        pdf_text, pdf_images = get_pdf_text_images(pdf_files)
                        raw_text += pdf_text
                        images.extend(pdf_images)
                    if ppt_files:
                        ppt_text, ppt_images = get_ppt_text_images(ppt_files)
                        raw_text += ppt_text
                        images.extend(ppt_images)
                    if docx_files:
                        raw_text += get_doc_text(docx_files)

                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.processed = True
                    st.session_state.images = images
                    st.success("Processing complete. You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF, PPT, or DOC document before processing.")

    if st.session_state.processed:
        question_col, clear_button_col = st.columns([0.8, 0.2])
        with question_col:
            user_question = st.text_input("Ask a question:", "", key="question", help="Type your question here")
        with clear_button_col:
            if st.button("Clear"):
                st.session_state.qa_pairs = []

        if user_question:
            with st.spinner('Fetching answer...'):
                answer = user_input(user_question, st.session_state.images)
                st.session_state.qa_pairs.append((user_question, answer))

        for question, answer in reversed(st.session_state.qa_pairs):
            st.markdown(f'<div class="question-box"><strong>Question:</strong><br>{question}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box"><strong>Answer:</strong><br>{answer}</div>', unsafe_allow_html=True)
    else:
        st.info("Please upload and process a document to start asking questions.")

    st.markdown('<div class="footer">©Arijeet Jash</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
