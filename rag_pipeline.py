from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import load_prompt
from langchain.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import pytesseract
from dotenv import load_dotenv
import cv2
import base64
import numpy as np
from prompt_template import template,response_schemas,parser

load_dotenv()

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

async def process_image(image):
    if not isinstance(image, str):
        print(f"[ERROR] Expected base64 string, but got: {type(image)}")
        return ""

    if image is None or image == "":
        return None

    try:
        image_bytes = base64.b64decode(image)
        if not image_bytes:
            return None

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_cv is None:
            return None

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        extracted_text = pytesseract.image_to_string(gray)
        return extracted_text.strip()
    except Exception as e:
        print(f"Image processing error : {e}")
        return ""


chat_model = GoogleGenerativeAI(model="gemini-1.5-flash",temperature = 0.2)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.load_local("tds_index", embedding_model, allow_dangerous_deserialization=True)
multiqueryretriver = MultiQueryRetriever.from_llm(
    retriever = vector_store.as_retriever(search_type="similarity",search_kwargs = {'k' : 4}),
    llm = chat_model
)

prompt = template
parallel_chain = RunnableParallel(
    context = multiqueryretriver | RunnableLambda(format_docs),
    question = RunnablePassthrough(),
    image_text=RunnableLambda(process_image)
)
main_chain = parallel_chain | prompt | chat_model | parser

async def generate_answer(question, image = None):
    result = await main_chain.ainvoke({
        "question": question,
        "image_text": image
    })

    try:
        return result  # return dict with "answer" and "links"
    except Exception as e:
        return {
            "Exception" : e,
            "answer": "Sorry, I couldn't format the response correctly.",
            "links": []
        }