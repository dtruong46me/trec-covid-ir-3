import os
import sys

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from weaviate.collections import Collection

__root__ = os.getcwd()
sys.path.insert(0, __root__)

load_dotenv()

from src.utils import load_topic_file


def generate_hyde(collection: Collection, query_text: str, limit: int, **kwargs) -> str:
    """
    Hypothetical Document Embeddings (HyDE) search
    Args:
        collection (Collection): Weaviate collection to search
        query_text (str): The search query text
        limit (int): Number of results to return
    Returns:
        str: Hypothetical document generated from the query
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    chat_model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY, # type: ignore
    )

    system_prompt = """You are an expert biomedical and scientific assistant.

Your task is to answer the given research question directly and concisely.

Guidelines:
- Start with a clear, direct answer to the question (1 sentence)
- Optionally add 1–2 sentences of brief explanation or clarification
- Use precise scientific or biomedical terminology when relevant
- Focus on facts, mechanisms, or consensus understanding
- Do NOT provide medical advice
- Do NOT mention uncertainty unless it is scientifically necessary
- Do NOT mention that this is a hypothetical answer
- Keep the total length under 80 words
"""

    human_message = f"""Query: {query_text}
Question: {query_text}

Provide a direct answer to the question.
"""
    
    response = chat_model.invoke(
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_message}
        ]
    )

    hypothetical_doc: str = response.content # type: ignore

    return hypothetical_doc


if __name__ == "__main__":
    topics_df = load_topic_file()
    test_topic = topics_df.iloc[0]

    query = test_topic['query']
    question = test_topic['question']

    query_text = f"Query: {query}\nQuestion: {question}"

    hyde_doc = generate_hyde(None, query_text, limit=1) # type: ignore

    print(f"[+] Original Query:\n{query_text}\n")
    print("[+] Generated Hypothetical Document:\n")
    print(hyde_doc)