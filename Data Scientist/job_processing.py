from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model_hf = HuggingFaceEmbeddings(
  model_name=EMBEDDING_MODEL_HF,
  model_kwargs={'device': 'cpu'}, # Additional options to specify that the model should run on CPU instead of GPU.
  encode_kwargs={'normalize_embeddings': True} # Additional options to specify that the embeddings should be normalized (converted to unit vectors) after encoding.
)

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Process job opening data from csv
def job_processing(file_path, meta_data_cols: list):
    """
    Load, clean, embed, and storing csv file to vector database.

    Arguments
        - file_path: CSV document of job listings
        - meta_data_cols: Column names that will be used for filtering 
    """
    loader_csv = CSVLoader(file_path, metadata_columns=meta_data_cols, encoding="utf-8", csv_args={"delimiter": ","})
    docs = loader_csv.load()
    all_contents = [post for post in docs]
    chunks = SPLITTER.split_documents(all_contents)
    
    db_dynamic = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model_hf,
        persist_directory="./chroma_db/dynamic_csv"
    )
    print("Stored to vector db using Hugging Face.")
        
    print(f"Dynamic index updated: {len(chunks)} chunks")