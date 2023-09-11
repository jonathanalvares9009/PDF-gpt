import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from pdf2image import convert_from_path
import tempfile, time, os


def create_images_from_pdf(pdf_docs):
    """Creates and saves images to output directory with the folder name as the file name"""
    for pdf_doc in pdf_docs:
       # Create a temporary PDF file
        with tempfile.NamedTemporaryFile() as f:
            f.write(pdf_doc.getbuffer())
            images = convert_from_path(f.name)

        # Check if the output directory exists
        if not os.path.exists("output"):
            os.mkdir("output")

        # Check if the folder exists
        if not os.path.exists(f"output/{pdf_doc.name}"):
            os.mkdir(f"output/{pdf_doc.name}")

        # Save the images to the output directory
        for idx, image in enumerate(images):
            image.save(f"output/{pdf_doc.name}/_{idx}.jpg", "JPEG")



def get_pdf_text(pdf_docs):
    """Extracts text from PDFs using OCR and returns a string"""
    text = ""

    # Save the PDFs to a temporary directory
    with tempfile.TemporaryDirectory() as path:
        for pdf_doc in pdf_docs:
            with open(f"{path}/{pdf_doc.name}", "wb") as f:
                f.write(pdf_doc.getbuffer())

        # Load the PDFs
        loader = UnstructuredPDFLoader(f"{path}/{pdf_doc.name}", max_partition=1000, strategy="ocr_only")
        data = loader.load()
        print("Loaded PDFs: ", data)


    return text




def get_text_chunks(raw_text):
    """Extracts text chunks from a string and returns a list of strings"""
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks


def get_vector_store(text_chunks):
    """Creates a vector store from a list of strings and returns a vector store"""
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store


def create_conversation_chain(vector_store):
    """Creates a conversation chain from a vector store and returns a conversation chain"""
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain


def handle_userquestion(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for idx, message in enumerate(st.session_state.chat_history):
        if idx % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    st.session_state.user_question = ""


def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input(
        "Ask questions about the PDFs here: ")

    if user_question:
        handle_userquestion(user_question)

    st.write(css, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                start_time = time.time()
                create_images_from_pdf(pdf_docs)
                print(f"Time taken for creating images: {time.time() - start_time} seconds")

                # #  Get the PDF text
                # start_time = time.time()
                # print("Getting PDF text")
                # raw_text = get_pdf_text(pdf_docs)
                # print(f"Time taken: {time.time() - start_time} seconds")

                # # Save the PDF to a text file
                # with open("pdf_text.txt", "w") as f:
                #     f.write(raw_text)


                # #  Extract text chunks
                # print("Extracting text chunks")
                # start_time = time.time()
                # text_chunks = get_text_chunks(raw_text)
                # print(f"Time taken: {time.time() - start_time} seconds")

                # #  Create vector store
                # print("Creating vector store")
                # start_time = time.time()
                # vector_store = get_vector_store(text_chunks)
                # print(f"Time taken: {time.time() - start_time} seconds")

                # print("Creating conversation chain")
                # start_time = time.time()
                # # Create a conversation chain
                # st.session_state.conversation = create_conversation_chain(
                #     vector_store)
                # print(f"Time taken: {time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
