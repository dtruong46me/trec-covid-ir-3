import os
import sys

import pandas as pd
from tqdm import tqdm
from weaviate.classes.query import MetadataQuery

__root__ = os.getcwd()
sys.path.insert(0, __root__)

from src.utils import load_topic_file, generate_submission
from src.components.weaviate_conn import connect_weaviate_local

def main():
    print(f"Generating dummy submission file...")

    TOP_K = 100

    wv_client = connect_weaviate_local()
    collection = wv_client.collections.get("TREC_COVID_No_Embeddings1226")

    # Print total objects in collection
    print(f"Total objects in collection '{collection.name}': {collection.aggregate.over_all(total_count=True).total_count}")

    topics_df = load_topic_file()

    results = []

    for idx, row in tqdm(topics_df.iterrows(), total=len(topics_df), desc="Generating results"):
        topic_id = row['topic-id']

        query_text = f"{row['query']} {row['question']} {row['narrative']}"

        # Get top N documents
        response = collection.query.bm25(
            query=query_text,
            limit=TOP_K,
            return_metadata=MetadataQuery(score=True) # Retrieve BM25 score
        )

        for obj in response.objects:
            results.append({
                "topic-id": topic_id,
                "cord-id": obj.properties['cord_uid']
            })

    results_df = pd.DataFrame(results)
    generate_submission(results_df)

if __name__ == "__main__":
    main()