from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pdf2image import convert_from_path
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os
import io
import openai
import asyncio
import fitz 
import nltk
from nltk.tokenize import sent_tokenize
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, ForeignKey, Float
from sqlalchemy.dialects.postgresql import ARRAY
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import text



from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()


BUCKET_NAME = os.getenv('NEXT_PUBLIC_S3_BUCKET_NAME')
UPLOAD_BUCKET_NAME = os.getenv('NEXT_PUBLIC_S3_IMAGE_BUCKET_NAME')
AWS_ACCESS_KEY_ID = os.getenv('NEXT_PUBLIC_S3_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('NEXT_PUBLIC_S3_SECRET_ACCESS_KEY')

DATABASE_URL = os.getenv("DATABASE_URL")


engine = create_engine(
    DATABASE_URL
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

nltk.download('punkt')

metadata = MetaData()

sourcebook_embedings = Table(
    'sourcebook_embedings', metadata,
    Column('id', Integer, primary_key=True),
    Column('sourcebook_hash', String, ForeignKey('sourcebooks.hash')),
    Column('vector', ARRAY(Float)),
    Column('content', String),
    Column('page_number', Integer),
)

def is_two_page_spread(image):
    width, height = image.size
    return width > height

def download_file_from_s3(bucket_name, file_name):
    try:
        s3_client.download_file(bucket_name, file_name, file_name)
        return True
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="Credentials not available")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))

def upload_file_to_s3(bucket_name, file_name, file):
    try:
        s3_client.upload_fileobj(file, bucket_name, file_name)
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="Credentials not available")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def embed_document(doc_content: str, hash: str, page_number: int) -> dict:
    # Call OpenAI API to get embeddings for the document content
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=doc_content.replace("\n", " ")
    )
    # Since the response object structure is known from the print statement:
    # Access the embeddings directly from the object
    embedding = response['data'][0]['embedding'] if isinstance(response, dict) else response.data[0].embedding

    return {
        "sourcebookHash": hash,
        "vector": embedding,
        "content": doc_content,
        "pageNumber": page_number,
    }

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def split_text_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

def split_text_into_segments(text, max_length=2000):
    sentences = split_text_into_sentences(text)
    segments = []
    current_segment = ""

    for sentence in sentences:
        if len(current_segment) + len(sentence) <= max_length:
            current_segment += (sentence + " ")
        else:
            segments.append(current_segment)
            current_segment = sentence + " "

    if current_segment:
        segments.append(current_segment)

    return segments

def format_vector_as_pg_array(vector):
    # Properly close the parentheses and ensure the array is formatted as PostgreSQL expects
    return '[' + ','.join(map(str, vector)) + ']'

async def insert_sourcebook_embedding(embedding_data):
    # Execute the query using the parameter dictionary
    # Construct the query with proper placeholders for using prepared statements

    print(embedding_data)

    query = text("""
    INSERT INTO sourcebook_embedings (sourcebook_hash, vector, content, page_number)
    VALUES (:sourcebook_hash, :vector, :content, :page_number)
    """)

    # Use a dictionary to pass parameters to avoid SQL injection and ensure proper type handling
    params = {
        "sourcebook_hash": embedding_data["sourcebookHash"],
        "vector": format_vector_as_pg_array(embedding_data["vector"]),
        "content": embedding_data["content"],
        "page_number": embedding_data["pageNumber"],
    }
    
    print(embedding_data["pageNumber"])
    
    with engine.connect() as con:
        con.execute(query, params)
        con.commit()

async def loadS3IntoPGVector(hash: str):
    # Download PDF from S3
    file_name = f"{hash}.pdf"
    
    # Process the PDF to extract text (placeholder function)
    pdf_text = extract_text_from_pdf(file_name)
    
    # Split text into segments
    segments = split_text_into_segments(pdf_text)
    
    # Process segments to get embeddings
    results = await asyncio.gather(*(embed_document(segment, hash, i) for i, segment in enumerate(segments)))
    
    # Insert data into the database
    for result in results:
        await insert_sourcebook_embedding(result)
    
    return results[0]  # Assuming this is the expected return value

@app.post("/process-pdf/")
async def process_pdf(hash: str = Form(...)):
    pdf_file_name = f"{hash}.pdf"
    output_file_name = f"{hash}.jpg"
    
    # Step 1: Download PDF from S3
    if not download_file_from_s3(BUCKET_NAME, pdf_file_name):
        raise HTTPException(status_code=500, detail="Failed to download PDF from S3")
    
    # Step 2: Convert PDF to Image
    images = convert_from_path(pdf_file_name)
    if not images:
        raise HTTPException(status_code=404, detail="No images found in the PDF")
    
    # Determine if cropping is needed and perform cropping if necessary
    img = images[0]
    if is_two_page_spread(img):
        width, height = img.size
        left = width / 2
        top = 0
        right = width
        bottom = height
        img = img.crop((left, top, right, bottom))

    # Save the processed image to a BytesIO object
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # Step 3: Upload to S3
    upload_file_to_s3(UPLOAD_BUCKET_NAME, output_file_name, img_byte_arr)

    await loadS3IntoPGVector(hash)

    return {"message": "PDF processed and image uploaded to S3", "image_name": output_file_name}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
