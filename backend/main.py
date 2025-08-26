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
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")


# ----------- Utilities ------------
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str, max_words: int = 1000):
    """Split text into word chunks."""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


def make_quiz(questions, context):
    """Convert generated questions into quiz format with options."""
    quiz = []
    for q in questions:
        try:
            # Get the correct answer
            ans = qa({"question": q, "context": context})["answer"]

            # Simple distractors (could be improved)
            distractors = [
                ans[::-1],  # reversed string as a fake option
                "None of the above",
                "I don't know"
            ]

            options = [ans] + distractors
        except Exception:
            options = ["Option A", "Option B", "Option C", "Option D"]
            ans = "Option A"

        quiz.append({
            "question": q,
            "options": options,
            "correct": ans
        })
    return quiz


# ----------- Routes ------------
@app.get("/")
def root():
    return {"message": "FastAPI is running!"}


@app.post("/process")
async def process_pdf(pdf: UploadFile = File(...)):
    # Save uploaded file locally
    file_path = "uploaded.pdf"
    with open(file_path, "wb") as f:
        f.write(await pdf.read())

    # Extract text
    text = extract_text_from_pdf(file_path)

    # Step 1: Chunk + summarize each piece
    chunks = list(chunk_text(text, max_words=1000))
    summaries = []
    for chunk in chunks:
        try:
            summaries.append(
                summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
            )
        except Exception:
            continue

    # Step 2: Final summary from all summaries
    combined_summary = " ".join(summaries)
    final_summary = summarizer(
        combined_summary, max_length=300, min_length=100, do_sample=False
    )[0]['summary_text']

    # Step 3: Generate questions from final summary
    q_out = qg("generate questions: " + final_summary, max_length=64, num_return_sequences=5, num_beams=5)
    questions = [q["generated_text"] for q in q_out]

    # Step 4: Convert questions into quiz format
    quiz = make_quiz(questions, final_summary)

    return {
        "summary": final_summary,
        "quiz": quiz
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
