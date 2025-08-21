from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import uvicorn
import fitz

app = FastAPI()

# Allow frontend to talk to backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Later restrict to React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HuggingFace models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qg = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


@app.get("/")
def root():
    return {"message": "FastAPI is running!"}

@app.post("/process")
async def process_pdf(pdf: UploadFile = File(...)):
    # Save file locally
    file_path = "uploaded.pdf"
    with open(file_path, "wb") as f:
        f.write(await pdf.read())

    # Extract text
    text = extract_text_from_pdf(file_path)

    # Summarize
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']

    # Generate questions
    q_out = qg("generate questions: " + summary, max_length=64, num_return_sequences=5)
    questions = [q["generated_text"] for q in q_out]

    return {"summary": summary, "questions": questions}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
