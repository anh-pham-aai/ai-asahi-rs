from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from typing import List, Dict, Any
import os
from config import (
    logger,
    RAG_OPENAI_API_KEY,
    RAG_OPENAI_BASEURL,
    RAG_OPENAI_PROXY,
    RAG_AZURE_OPENAI_API_KEY,
    RAG_AZURE_OPENAI_ENDPOINT,
    RAG_AZURE_OPENAI_API_VERSION,
    retriever,
    get_env_variable,
    EmbeddingsProvider,
)

# Get LLM configuration from environment
LLM_PROVIDER = get_env_variable("LLM_PROVIDER", "openai").lower()
LLM_MODEL = get_env_variable("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(get_env_variable("LLM_TEMPERATURE", "0.0"))

# RAG prompt template
RAG_TEMPLATE = """You are an AI assistant for answering questions based on specific documents.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. DO NOT make up an answer.
If the context doesn't contain relevant information to the question, just say that you don't have enough information to answer.

Context:
{context}

Question: {question}

Answer:"""

def format_docs(docs: List[Document]) -> str:
    """Format a list of document objects into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])

def get_llm():
    """Initialize the appropriate LLM based on configuration."""
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=RAG_OPENAI_API_KEY,
            openai_api_base=RAG_OPENAI_BASEURL,
            openai_proxy=RAG_OPENAI_PROXY,
        )
    elif LLM_PROVIDER == "azure":
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            azure_deployment=LLM_MODEL,
            api_key=RAG_AZURE_OPENAI_API_KEY,
            azure_endpoint=RAG_AZURE_OPENAI_ENDPOINT,
            api_version=RAG_AZURE_OPENAI_API_VERSION,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

def create_rag_chain():
    """Create a RAG chain using LangChain."""
    # Initialize the language model
    llm = get_llm()
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

async def query_rag(question: str, file_ids: List[str] = None, k: int = 4) -> Dict[str, Any]:
    """
    Query the RAG system with a question and optional file_ids filter.
    
    Args:
        question: The user's question
        file_ids: Optional list of file IDs to restrict the search to
        k: Number of documents to retrieve
    
    Returns:
        Dictionary containing the answer and retrieved documents
    """
    try:
        # Create a filtered retriever if file_ids are provided
        if file_ids and len(file_ids) > 0:
            filtered_retriever = retriever.with_filter({"file_id": {"$in": file_ids}})
            filtered_retriever = filtered_retriever.with_k(k)
            
            # Get documents for context
            docs = await filtered_retriever.ainvoke(question)
        else:
            # Use the default retriever
            retriever_with_k = retriever.with_k(k)
            docs = await retriever_with_k.ainvoke(question)
        
        # If no documents were retrieved, return early
        if not docs:
            return {
                "answer": "I don't have any relevant information to answer your question.",
                "source_documents": [],
                "file_ids": []
            }
        
        # Create and run the RAG chain
        rag_chain = create_rag_chain()
        answer = await rag_chain.ainvoke(question)
        
        # Extract file_ids from the retrieved documents
        retrieved_file_ids = list(set(doc.metadata.get("file_id") for doc in docs if "file_id" in doc.metadata))
        
        # Format the response
        return {
            "answer": answer,
            "source_documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs
            ],
            "file_ids": retrieved_file_ids
        }
    
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        raise e 