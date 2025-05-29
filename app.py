# Required Importing Libraries
from uuid import uuid4
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from youtube_transcript_api import YouTubeTranscriptApi
import re
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import os

# LangChain and RAG tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# App Configuration
app = Flask(__name__)
app.secret_key = "aR4nd0mStr1ngW1th$peci@lChar$"
load_dotenv()

# Models
google_api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
parser = StrOutputParser()

# Prompt Template
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=['context', 'question']
)

# Format Documents
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Transcript Utilities
def get_transcript_languages(url_or_id):
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url_or_id)
    video_id = video_id_match.group(1) if video_id_match else url_or_id.split('&')[0]
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        return [{"language": t.language, "code": t.language_code} for t in transcripts]
    except Exception:
        return []

def fetch_transcript_text(video_id, language_code):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])
        return "\n".join([item['text'] for item in transcript_list])
    except Exception:
        return None

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/transcripts')
def transcripts():
    video = request.args.get('video', '')
    languages = get_transcript_languages(video)
    return jsonify({"languages": languages})

@app.route('/home/rag', methods=['GET', 'POST'])
def handle_rag_submission():
    global rag_chain
    video_url = request.form.get("video_url")
    language_code = request.form.get("language_code")

    # Extract video ID
    parsed_url = urlparse(video_url)
    video_id = None
    if "youtu.be" in parsed_url.netloc:
        video_id = parsed_url.path.strip("/")
    elif "youtube.com" in parsed_url.netloc:
        if parsed_url.path == "/watch":
            query_params = parse_qs(parsed_url.query)
            video_id = query_params.get("v", [None])[0]
        elif parsed_url.path.startswith("/embed/"):
            video_id = parsed_url.path.split("/embed/")[1]
    if not video_id:
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$)", video_url)
        video_id = match.group(1) if match else None

    if not video_id:
        return "Invalid YouTube URL. Could not extract video ID.", 400

    text = fetch_transcript_text(video_id, language_code)
    if text is None:
        return "Failed to fetch transcript.", 400

    docs = text_splitter.create_documents([text])
    vector_store = FAISS.from_documents(docs, embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", kwargs={"k": 5})

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    rag_chain = parallel_chain | prompt | model | parser
    return redirect(url_for('ask_question_page'))

@app.route('/ask', methods=['GET', 'POST'])
def ask_question_page():
    global rag_chain
    if not rag_chain:
        return "No RAG chain found. Please upload a video first.", 400

    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({"error": "No question provided."}), 400

        response = rag_chain.invoke(question)
        return jsonify({"question": question, "answer": response})

    return render_template('result.html')

# ------------------------- MAIN ENTRY POINT -------------------------
if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 10000))
    serve(app, host="0.0.0.0", port=port)