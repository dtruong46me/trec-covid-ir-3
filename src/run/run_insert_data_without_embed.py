import os
import sys
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from langchain_core.documents import Document

__root__ = os.getcwd()
sys.path.insert(0, __root__)

from src.components.data_loader import WeaviateDataLoader
from src.components.weaviate_conn import connect_weaviate_local

def process_dataframe_chunk(df_chunk: pd.DataFrame) -> List[Document]:
    """
    Process a DataFrame chunk into a list of Documents.
    Args:
        df_chunk (pd.DataFrame): A chunk of the DataFrame to process.
    Returns:
        List[Document]: A list of Document objects created from the DataFrame chunk.
    """
    metadata_list = df_chunk.to_dict('records')

    documents_chunk = [
        Document(
            page_content=f"{meta.pop('title', '')}\n{meta.pop('abstract', '')}",
            metadata=meta
        )
        for meta in metadata_list
    ]
    return documents_chunk

def process_df_multiprocessing(df: pd.DataFrame) -> List[Document]:
    """
    Load data from CSV, process it into Documents using multiprocessing,
    Args:
        df (pd.DataFrame): The DataFrame containing the data to process.
    Returns:
        List[Document]: A list of Document objects created from the DataFrame.
    
    """
    print("Converting DataFrame to List[Document] using multiprocessing...")
    
    # Get number of CPU cores, minus one to avoid system overload
    num_workers = os.cpu_count()
    max_workers = num_workers - 1 if num_workers and num_workers > 1 else 1 
    print(f"Using {max_workers} worker processes.")

    # Split the DataFrame into chunks for processing
    df_chunks = np.array_split(df, max_workers * 2) # Split into smaller chunks

    documents = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit the df chunks to worker processes for processing 
        futures = [executor.submit(process_dataframe_chunk, chunk) for chunk in df_chunks] # type: ignore
        
        # Receive results (with tqdm progress bar)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            documents.extend(future.result()) # Collect results from processes

    print(f"Conversion complete. Total documents processed: {len(documents)}")
    
    return documents


def main():
    
    COLLECTION_NAME = "TREC_COVID_No_Embeddings1226"
    DATA_PATH = os.path.join(__root__, "data", "raw", "CORD_19", "metadata.csv")

    schema = {
        "cord_uid": str,
        "title": str,
        "abstract": str,
        "pdf_json_files": str,
        "pmc_json_files": str,
    }

    df = pd.read_csv(DATA_PATH)
    df = df.replace({pd.NA: None})  # type: ignore
    df = df[["cord_uid", "title", "abstract", "pdf_json_files", "pmc_json_files"]]  # Select only necessary columns

    weaviate_client = connect_weaviate_local()

    wv_data_loader = WeaviateDataLoader(
        client=weaviate_client,
        embed_model=None # type: ignore
    )

    collection = wv_data_loader.init_collection(
        collection_name=COLLECTION_NAME,
        schema=schema
    )

    documents = process_df_multiprocessing(df=df)

    # documents = documents[:1000]  # Limit for testing

    wv_data_loader.insert_data_no_embedding(
        collection=collection,
        documents=documents
    )

if __name__ == "__main__":
    main()