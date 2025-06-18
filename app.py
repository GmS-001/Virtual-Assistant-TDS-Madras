from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from rag_pipeline import generate_answer
from typing import Optional
from pydantic import BaseModel,Field
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify domain(s) if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class QuestionRequest(BaseModel):
    question: str = Field(..., description="Ask questions related to the TDS course.")
    image: Optional[str] = Field(default=None, description="Base64 encoded image string")

@app.get('/')
def home() :
    return {'message' : 'AI Assistant for TDS Course - API.'}

@app.get("/about")
def about() :
    return {'message' : 'A fully fucntional API for asking any queries or doubts related to Tools in Data Science Course of IIT Madras.'}

@app.post('/ask')
async def ask(request: QuestionRequest) :
    if not request.question or not request.question.strip():
      raise HTTPException(status_code=400, detail="Question cannot be empty.")

    
    answer = await generate_answer(request.question,request.image)
    if answer :
        return JSONResponse(status_code=200, content= answer)
    else :
        raise HTTPException(status_code = 204, detail = 'Assistant is unable to answer.')


