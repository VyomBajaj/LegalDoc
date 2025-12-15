# backend/summarizer.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.pydantic import PydanticOutputParser
from models import LegalDocSummary, LastDateResponse
from datetime import date
from pydantic import BaseModel, Field
from typing import Optional

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

def get_model(api_key: str):
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.0,
        max_output_tokens=8192,
        google_api_key=api_key
    )

def generate_document_summary(document_content: str, language: str, google_api_key: str) -> LegalDocSummary:
    # ... (existing code for summarization) ...
    model = get_model(google_api_key)
    parser = PydanticOutputParser(pydantic_object=LegalDocSummary)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert legal assistant. Summarize the following legal document in a structured format."),
        ("human", "Summarize the following document in {language}. "
         "The document content is: \n\n{document_content}\n\n"
         "{format_instructions}"
        )
    ]).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser
    
    try:
        summary = chain.invoke({
            "language": language,
            "document_content": document_content
        })
        return summary
    except Exception as e:
        return LegalDocSummary(
            category="Error",
            description="Could not generate a summary due to an error.",
            important_timeline=[],
            main_takeaway=[f"An error occurred: {str(e)}"],
            risk_factors=[],
            next_steps=[],
            urgency_percentage=0,
            urgency_level="Low"
        )


# New function for date extraction
def extract_last_date(document_content: str, google_api_key: str) -> Optional[date]:
    """
    Extracts the last date to take action from a document.
    """
    model = get_model(google_api_key)
    
    class LastDateExtractor(BaseModel):
        last_date: Optional[date] = Field(None, description="The last date to take action, in YYYY-MM-DD format. Return null if not found.")

    parser = PydanticOutputParser(pydantic_object=LastDateExtractor)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert legal assistant. Extract the final deadline or last date for action mentioned in the document. The date should be in YYYY-MM-DD format."),
        ("human", "Document content: \n\n{document_content}\n\n{format_instructions}")
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser
    
    try:
        response = chain.invoke({"document_content": document_content})
        return response.last_date
    except Exception as e:
        print(f"Error extracting date: {e}")
        return None
