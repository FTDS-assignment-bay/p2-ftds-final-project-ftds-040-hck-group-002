# Import libraries
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

from pdfminer.high_level import extract_text
import re
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from io import BytesIO
import json

# Load the `.env` file that contains secret variables like API Keys
load_dotenv()

# We set the Google API Key for authentication when using Google's AI models. This key is necessary to access the services and is kept secret to prevent unauthorized use.
# Ensure you keep the API Key secret in real projects to prevent others from stealing your quota.
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

# General variable
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
CHAT_MODEL = "gemini-3.1-flash-lite-preview"
EMBEDDING_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2"
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)

# Embedding
## Using Gemini
embedding_model_gemini = GoogleGenerativeAIEmbeddings(
    google_api = GEMINI_API_KEY,
    model = EMBEDDING_MODEL
)

## Using Hugging Face (alternative)
embedding_model_hf = HuggingFaceEmbeddings(
  model_name=EMBEDDING_MODEL_HF,
  model_kwargs={'device': 'cpu'}, # Additional options to specify that the model should run on CPU instead of GPU.
  encode_kwargs={'normalize_embeddings': True} # Additional options to specify that the embeddings should be normalized (converted to unit vectors) after encoding.
)

chat_model = ChatGoogleGenerativeAI(
            google_api_key=GEMINI_API_KEY,
            model=CHAT_MODEL
        )

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

def extract_cv(uploaded_file) -> str:
    """
    Extract CV from PDF to text.

    Arguments
        uploaded: file uploaded in streamlit.
    Return
        cv_text: CV in form of text
    """
    loader = extract_text(BytesIO(uploaded_file.getvalue())) # Load file

    # Clean file
    cv_text = loader.replace("\xa0", " ") # Remove non-breaking space
    cv_text = "\n".join(line.rstrip() for line in cv_text.splitlines()) # Remove trailing whitespace for each row
    cv_text = re.sub(r"\n{3,}", "\n\n", cv_text) # Two new lines maximum

    return cv_text

# Match pipeline
def match_cv_to_jobs(cv_path: str, value_to_extract: str):
    """
    Process CV and job vacancy dataset, then matches it. Saves the result in a form of json then saves it accordingly.

    Arguments:
    - cv_path: file path of CV in pdf format.
    - value_to_extract: intended position that will be filtered. 
    """
    cv_profile = extract_cv(cv_path)
            
    # Load index  
    dynamic_store = Chroma(
        persist_directory="./chroma_db/dynamic_csv",
        embedding_function=embedding_model_hf
    )

    # Find matching job
    query = f"""
        Find at least two of the most suitable job post based on candidate's skills and experience for
        Role:
        {value_to_extract}

        Profile candidate:
        {cv_profile}
    """
    job_matches = dynamic_store.similarity_search(query=query, k=5, filter={"job_category": value_to_extract})
    jobs_text = "\n\n---\n\n".join([d.page_content for d in job_matches]) # Formatting
    
    prompt = PromptTemplate.from_template("""
        You are an AI Career Advisor specializing in resume evaluation and job matching.

        Your task is to analyze the candidate's CV against the retrieved job vacancies.

        Instructions:
        - Return ONLY a valid JSON object.
        - Do NOT wrap the response inside markdown or 'json'.
        - Do NOT add any explanation before or after the JSON.
        - Follow the schema exactly.
        - Do NOT omit any key.
        - If information is unavailable, fill with Unknown.
        - Integer fields must contain only numbers.
        - Percentage values must be integers between 0 and 100.
        - Arrays must always exist, even if empty.
        - Return 'top_jobs' from retrieved Job Vacancies ONLY.
        - Try to always return data based on existing knowledge of cv_profile and job_listings.
        - Return 'cv_quality' based on 'cv_profile' with Application Tracking System (ATS) scoring method.

        Candidate Profile:
        {cv_profile}

        Retrieved Job Vacancies:
        {job_listings}

        Return this schema exactly:

        {{
            "candidate": {{
                "name": string,
                "target_role": {value_to_extract},
                "professional_summary": string
            }},

            "scores": {{
                "cv_quality": integer (1-100),
                "job_relevance": integer (1-100),
                "overall_readiness": integer (1-100)
            }},

            "salary_prediction": {{
                "currency": string,
                "average": integer (in IDR),
                "minimum": integer (in IDR),
                "maximum": integer (in IDR)
            }},

            "top_jobs": [
                {{
                    "rank": integer,
                    "job_title": string,
                    "company": string,
                    "industry": string,
                    "location": string,
                    "match_score": integer (1-100),
                    "estimated_salary": integer (in IDR),
                    "reason": string
                }}
            ],

            "skill_analysis": {{
                "matched_skills": [
                    {{
                        "skill": string,
                        "score": integer (1-100)
                    }}
                ],

                "missing_skills": [
                    string
                ],

                "market_demand_skills": [
                    {{
                        "skill": string,
                        "demand_score": integer
                    }}
                ]
            }},

            "recommendations": [
                {{
                    "priority": string,
                    "category": string,
                    "title": string,
                    "description": string
                }}
            ]
        }}
    """)

    chain = prompt | chat_model
    response = chain.invoke({
        "cv_profile": cv_profile,
        "job_listings": jobs_text,
        "value_to_extract": value_to_extract
    })
    
    response_json = "".join(
                part["text"]
                for part in response.content
                if part["type"] == "text"
            )
    
    data = json.loads(response_json)

    with open("output.json", "w") as file:
        json.dump(data, file, indent=4)