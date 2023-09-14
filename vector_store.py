from transformers import pipeline
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from multiprocessing import Pool, Value

def get_vector_store():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="vector_store", embedding_function=embedding_function)

def get_indices_of_start_batch(num_images):
    """Returns a list of indices to start the batch. Each batch should have 10 or less elements"""
    indices = []
    for i in range(0, num_images, 10):
        indices.append(i)
    indices.append(num_images)
    return indices

def summarize_text(text):
    summariser = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", tokenizer="sshleifer/distilbart-cnn-12-6")
    response = summariser(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
    return response

def embbed_vectors(text_chunks):
    """Creates a vector store from a list of strings and returns a vector store"""
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Loop through the text chunks and create a vector store
    db = Chroma(persist_directory="vector_store", embedding_function=embedding_function)
    for idx, text_chunk in enumerate(text_chunks):
        # Sumarize the text chunk
        summary = summarize_text(text_chunk)
        # Add an emdedding to the vector store and summary as the metadata
        db.add_texts(texts=[text_chunk], metadata=[summary])

    print("Processing finished")

def create_vector_store(text_chunks):
    start_batch_indices = get_indices_of_start_batch(len(text_chunks))
    pool = Pool(processes=6)

    for i in range(len(start_batch_indices) - 1):
        start_idx = start_batch_indices[i]
        end_idx = start_batch_indices[i + 1]
        pool.apply_async(embbed_vectors, args=[text_chunks[start_idx: end_idx]], error_callback=lambda e: print(e))

    pool.close()
    pool.join() 

if __name__ == "__main__":
    text_chunks = ["text chunk 1", "text chunk 2", "text chunk 3"]
    create_vector_store(text_chunks)