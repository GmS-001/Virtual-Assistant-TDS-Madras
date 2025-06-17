from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define schema for response
response_schemas = [
    ResponseSchema(
        name="answer",
        description="A clear, concise answer to the student's question."
    ),
    ResponseSchema(
        name="links",
        description="A list of up to 3 relevant forum links. Each link should be a dictionary with 'url' and 'text'."
    )
]

# Create the parser
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create the prompt template
template = PromptTemplate(
    template="""
You are the Virtual Teaching Assistant for the “Tools in Data Science (TDS)” course at IIT Madras.

You have access to:
• Course materials and official forum discussions (provided as “Context”).
• Any text extracted from student uploaded images (provided as “Image_Text”).

Your goal is to answer student questions with:
1. Accurate, concise explanations.
2. Direct references to Context (quote or summarize).
3. Incorporation of Image_Text when relevant.

If something isn’t covered by Context or Image_Text, respond:
“I’m sorry, I don’t have enough information to answer that from the provided materials.”

Keep a friendly, peer-to-peer tone.

**Question:**  
{question}

**Image Extract (if any):**  
{image_text}

**Relevant Context:**  
{context}

Respond in **strict** JSON format as per the following instructions:
{format_instructions}

Do not add anything outside this JSON format. If no links are relevant, return an empty array for "links".
""",
    input_variables=['context', 'question', 'image_text'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# Save template
# template.save('template.json')
