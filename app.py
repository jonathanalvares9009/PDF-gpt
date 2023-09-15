import streamlit as st
# from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from pdf2image import convert_from_path
import tempfile, time, os
from ocr import process_images
from vector_store import create_vector_store, get_vector_store


def create_images_from_pdf(pdf_doc, folder_name):
    """Creates and saves images to output directory with the folder name as the file name"""
    # Check if the output folder has a directory with the same name as the PDF
    if os.path.exists(f"output/images/{folder_name}"):
        return

    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile() as f:
        f.write(pdf_doc.getbuffer())
        images = convert_from_path(f.name)

    # Check if the output directory exists
    if not os.path.exists("output"):
        os.mkdir("output")

    # Check if the images folder exists
    if not os.path.exists("output/images"):
        os.mkdir("output/images")

    folder_name = pdf_doc.name.split(".")[0]

    # Check if the folder exists
    if not os.path.exists(f"output/images/{folder_name}"):
        os.mkdir(f"output/images/{folder_name}")

    # Save the images to the output directory
    for idx, image in enumerate(images):
        image.save(f"output/images/{folder_name}/_{idx}.jpg", "JPEG")


def get_text_chunks(folder_name):
    """Extracts text chunks from a string and returns a list of strings"""
    raw_ocr = ""

    # Check if the output folder has a directory with the same name as the PDF
    if not os.path.exists(f"output/text/{folder_name}"):
        return
    
    # Read all the files in output/text/{folder_name}
    for file_name in os.listdir(f"output/text/{folder_name}"):
        with open(f"output/text/{folder_name}/{file_name}") as f:
            raw_ocr += f.read()

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = text_splitter.split_text(raw_ocr)
    return text_chunks


def create_conversation_chain(vector_store):
    """Creates a conversation chain from a vector store and returns a conversation chain"""
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain


def handle_userquestion(user_question):
    """Handles a user question and returns a response"""
    if st.session_state.conversation:
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
    else:
        st.write("Please process the PDFs first")


def main():
    # load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(page_title="Chat with your Compiler Design Tutor",
                       page_icon=":books:")
    st.header("Chat with tutor :teacher:")
    user_question = st.text_input(
        "Ask questions about the PDFs here: ")

    if user_question:
        handle_userquestion(user_question)

    st.write(css, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_doc = None
        if st.button("Upload PDFs"):
            pdf_doc = st.file_uploader(
                "Upload your PDFs here")
        if st.button("Start"):
            with st.spinner("Processing PDFs..."):
                if pdf_doc:
                    folder_name = pdf_doc.name.split(".")[0]

                    start_time = time.time()
                    create_images_from_pdf(pdf_doc, folder_name)
                    print(f"Time taken for creating images: {time.time() - start_time} seconds")

                    start_time = time.time()
                    process_images(folder_name)
                    print(f"Time taken for creating OCR: {time.time() - start_time} seconds")

                    start_time = time.time()
                    text_chunks = get_text_chunks(folder_name)
                    print(f"Time taken for creating text chunks: {time.time() - start_time} seconds")    

                    # Create vector store if it doesn't exist
                    start_time = time.time()
                    create_vector_store(text_chunks)
                    print(f"Time taken for creating a vector store: {time.time() - start_time} seconds")

                # Create a conversation chain
                start_time = time.time()
                vector_store = get_vector_store()
                st.session_state.conversation = create_conversation_chain(
                    vector_store)
                print(f"Time taken for creating a conversation chain: {time.time() - start_time} seconds")
                
                # docs = vector_store.max_marginal_relevance_search(query="What is the name of the first chapter?", k=5)
                # for doc in docs:
                #     st.write(doc)


if __name__ == "__main__":
    main()
