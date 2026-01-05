import os
import sys
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from weaviate.classes.query import MetadataQuery
from weaviate.collections.classes.internal import QueryReturn
from weaviate.collections import Collection
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_voyageai.rerank import VoyageAIRerank
from langchain_core.documents import Document

__root__ = os.getcwd()
sys.path.insert(0, __root__)

from src.utils import load_topic_file, generate_submission
from src.components.weaviate_conn import connect_weaviate_local

load_dotenv()


# ==========================================
# SEARCH STRATEGIES
# ==========================================

def search_bm25(collection: Collection, query_text: str, limit: int, **kwargs) -> QueryReturn:
    """
    BM25 keyword-based search
    Args:
        collection (Collection): Weaviate collection to search
        query_text (str): The search query text
        limit (int): Number of results to return
    Returns:
        QueryReturn: Search results from Weaviate
    """
    return collection.query.bm25(
        query=query_text,
        limit=limit,
        return_metadata=MetadataQuery(score=True)
    )


def search_vector(collection: Collection, query_text: str, query_vector: List[float], limit: int, **kwargs) -> QueryReturn:
    """
    Pure vector similarity search
    Args:
        collection (Collection): Weaviate collection to search
        query_text (str): The search query text
        query_vector (List[float]): The query vector for similarity search
        limit (int): Number of results to return
    Returns:
        QueryReturn: Search results from Weaviate
    """
    return collection.query.near_vector(
        near_vector=query_vector,
        limit=limit,
        return_metadata=MetadataQuery(distance=True)
    )


def search_hybrid(collection: Collection, query_text: str, query_vector: List[float], limit: int, alpha: float = 0.5, **kwargs) -> QueryReturn:
    """
    Hybrid search combining BM25 and vector search
    Args:
        collection (Collection): Weaviate collection to search
        query_text (str): The search query text
        query_vector (List[float]): The query vector for similarity search
        limit (int): Number of results to return
        alpha (float): Weighting factor between BM25 and vector search (0.0=BM25, 1.0=vector)
    Returns:
        QueryReturn: Search results from Weaviate
    """
    return collection.query.hybrid(
        query=query_text,
        vector=query_vector,
        alpha=alpha,  # 0.0 = pure BM25, 1.0 = pure vector, 0.5 = balanced
        limit=limit,
        return_metadata=MetadataQuery(score=True)
    )


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


# Mapping search method names to functions
SEARCH_METHODS = {
    "bm25": search_bm25,
    "vector": search_vector,
    "hybrid": search_hybrid,
}


# ==========================================
# MAIN PIPELINE
# ==========================================

def run_retrieval_pipeline(
    search_method: str = "hybrid",
    alpha: float = 0.5,
    top_k: int = 100,
    top_k_rerank: int = 20,
    use_reranker: bool = True,
    use_hyde: bool = False,
    collection_name: str = "TREC_COVID_OpenAIEmbed_small"
):
    """
    Run the retrieval pipeline with configurable search method.
    Args:
        search_method: One of ["bm25", "vector", "hybrid"]
        alpha: Alpha parameter for hybrid search (0.0=BM25, 1.0=vector)
        top_k: Number of documents to retrieve in first stage
        top_k_rerank: Number of documents after reranking
        use_reranker: Whether to use Voyage AI reranker
        collection_name: Name of Weaviate collection
    """
    print(f"=== Retrieval Pipeline ===")
    print(f"Search Method: {search_method}")
    if search_method == "hybrid":
        print(f"Alpha: {alpha}")
    print(f"Top-K: {top_k}")
    print(f"Reranker: {'Enabled' if use_reranker else 'Disabled'} (Top-{top_k_rerank})")
    print(f"Use HyDE: {'Enabled' if use_hyde else 'Disabled'}")
    print(f"Collection: {collection_name}")
    print("=" * 50)

    # Validate search method
    if search_method not in SEARCH_METHODS:
        raise ValueError(f"Invalid search method: {search_method}. Choose from {list(SEARCH_METHODS.keys())}")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

    wv_client = connect_weaviate_local()
    collection = wv_client.collections.get(collection_name)

    openai_embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY, # type: ignore
    )

    reranker = None
    if use_reranker:
        reranker = VoyageAIRerank(
            voyage_api_key=VOYAGE_API_KEY, # type: ignore
            model="rerank-2.5",
            top_k=top_k_rerank
        )

    # Print collection stats
    print(f"Total objects in collection '{collection.name}': {collection.aggregate.over_all(total_count=True).total_count}")

    topics_df = load_topic_file()
    hypothetical_dir = os.path.join(__root__, "output", "hypothetical_documents", "run_20260105_093719.csv")
    hypothetical_docs_df: pd.DataFrame = pd.read_csv(hypothetical_dir)

    results = []
    hypothetical_docs = {}

    # Select search function
    search_fn = SEARCH_METHODS[search_method]

    for _, row in tqdm(topics_df.iterrows(), total=len(topics_df), desc="Processing queries"):
        topic_id = row['topic-id']
        query_text = f"{row['query']} {row['question']}"
        hypothetical_document = hypothetical_docs_df.loc[hypothetical_docs_df['topic-id'] == topic_id, 'hypothetical_document']
        query_text = query_text + "\n" + hypothetical_document.values[0] # type: ignore

        # Prepare search parameters
        search_params = {
            "collection": collection,
            "query_text": query_text,
            "limit": top_k,
        }

        # Generate hypothetical document if using hyde method
        if use_hyde:
            query_text_new = f"Query: {row['query']}\nQuestion: {row['question']}"
            hypothetical_doc = generate_hyde(collection=collection, query_text=query_text_new, limit=top_k)
            query_text = query_text + "\n" + hypothetical_doc
            hypothetical_docs[topic_id] = hypothetical_doc
        
        # Add vector if needed
        if search_method in ["vector", "hybrid"]:
            query_vector = openai_embedding_model.embed_query(query_text)
            search_params["query_vector"] = query_vector
        
        # Add alpha if hybrid
        if search_method == "hybrid":
            search_params["alpha"] = alpha

        # Execute search
        response: QueryReturn = search_fn(**search_params)

        # Process results
        if use_reranker and reranker:
            # Prepare candidates for reranking
            rerank_query = f"{row['query']} {row['question']} {row['narrative']}"
            candidates = []
            doc_map = {}

            for idx, obj in enumerate(response.objects):
                doc_text = f"{obj.properties['title']} {obj.properties['abstract']}"
                candidates.append(Document(page_content=doc_text, metadata=obj.properties))
                doc_map[idx] = obj.properties['cord_uid']

            if candidates:
                reranked_results = reranker._rerank(
                    documents=candidates,
                    query=rerank_query
                )

                # Map reranked indices back to cord_uid
                for ranked_doc in reranked_results.results:
                    results.append({
                        "topic-id": topic_id,
                        "cord-id": doc_map[ranked_doc.index]
                    })
        else:
            # No reranking - use search results directly
            for obj in response.objects:
                results.append({
                    "topic-id": topic_id,
                    "cord-id": obj.properties['cord_uid']
                })

    # Save hypothetical documents if generated
    if hypothetical_docs:
        from datetime import datetime
        output_dir = os.path.join(__root__, "output", "hypothetical_documents")
        os.makedirs(output_dir, exist_ok=True) # type: ignore
        output_filename = os.path.join(output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        hyde_df = pd.DataFrame(list(hypothetical_docs.items()), columns=['topic-id', 'hypothetical_document'])
        hyde_df.to_csv(output_filename, index=False)
        print(f"\n✓ Hypothetical documents saved to {output_filename}")

    # Generate submission file
    results_df = pd.DataFrame(results)
    generate_submission(results_df)
    print(f"\n✓ Submission file generated successfully!")


class SearchConfig(BaseModel):
    search_method: Literal["bm25", "vector", "hybrid"] = Field(
        description="Search method to use", 
        default="hybrid"
    )
    alpha: float = Field(
        description="Alpha parameter for hybrid search (0.0=BM25, 1.0=vector)",
        default=0.5
    )
    top_k: int = Field(
        description="Number of documents to retrieve",
        default=100
    )
    top_k_rerank: int = Field(
        description="Number of documents after reranking",
        default=20
    )
    use_reranker: bool = Field(
        description="Enable/disable Voyage AI reranker",
        default=True
    )
    collection_name: str = Field(
        description="Name of the collection to search",
        default="TREC_COVID_OpenAIEmbed_small"
    )
    use_hyde: bool = Field(
        description="Enable/disable Hypothetical Document Embeddings (HyDE)",
        default=False
    )


def main():
    # ==========================================
    # CONFIGURATION
    # ==========================================

    search_config = SearchConfig(
        search_method="hybrid",
        alpha=0.3,
        top_k=100,
        top_k_rerank=20,
        use_reranker=False,
        collection_name="TREC_COVID_OpenAIEmbed_small",
        use_hyde=False,
    )

    # Convert search_config into dictionary
    config_dict = search_config.model_dump()

    run_retrieval_pipeline(**config_dict)


if __name__ == "__main__":
    main()