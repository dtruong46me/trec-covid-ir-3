from typing import List, Type

import weaviate.classes as wvc
from langchain_core.documents import Document
from langchain_core.embeddings.embeddings import Embeddings
from tqdm import tqdm
from weaviate import WeaviateClient
from weaviate.collections import Collection
from weaviate.classes.config import Configure, Property, VectorDistances, DataType, Tokenization

class WeaviateDataLoader:
    def __init__(self, client: WeaviateClient, embed_model: Embeddings):
        self.client = client
        self.embed_model = embed_model

    @staticmethod
    def to_weaviate_datatype(dtype: Type) -> DataType:
        """
        (# <<< ĐÃ SỬA: Chuyển hàm này thành staticmethod để có thể gọi được)
        Ánh xạ kiểu dữ liệu Python sang kiểu dữ liệu của Weaviate.
        """
        mapping = {
            str: DataType.TEXT,
            int: DataType.INT,
            float: DataType.NUMBER,
            bool: DataType.BOOL,
            # Thêm các kiểu dữ liệu khác nếu cần
        }
        # Mặc định là TEXT nếu không tìm thấy
        return mapping.get(dtype, DataType.TEXT)

    def init_collection(self, collection_name: str, schema: dict) -> Collection:
        """
        Initialize a Weaviate collection with the given schema.
        Args:
            collection_name (str): The name of the collection to initialize.
            schema (dict): The schema definition for the collection, which keys are "properties" and values are "data types".
        Returns:
            Collection: An instance of the Collection class representing the initialized collection.
         """
        
        existing_collections = [col_name.lower() for col_name in self.client.collections.list_all().keys()]
        if collection_name.lower() in existing_collections:
            human_input = input(f"Collection '{collection_name}' already exists. Do you want to overwrite it? ([y]/n): ")
            if human_input.lower() in ["", "y", "yes"]:
                self.client.collections.delete(collection_name)
            else:
                return self.client.collections.get(collection_name)
    
        # Parse schema into Weaviate format (properties)
        weaviate_properties = [
            Property(name=key, data_type=self.to_weaviate_datatype(value))
            for key, value in schema.items()
        ]

        print(f"Creating collection '{collection_name}' with properties: {weaviate_properties}")
        collection = self.client.collections.create(
            name=collection_name,
            properties=weaviate_properties,
            inverted_index_config=Configure.inverted_index(
                index_null_state=True,
                index_property_length=True,
                index_timestamps=True,
                stopwords_removals=[],
                bm25_k1=1.2,
                bm25_b=0.75,
            )
        )

        return collection

    def insert_data(self, collection: Collection, documents: List[Document], batch_size: int = 128) -> None:
        """
        (# <<< THAY ĐỔI HOÀN TOÀN: Tối ưu hóa tốc độ insert)
        Insert tài liệu vào Weaviate sử dụng batch embedding và dynamic batching của Weaviate.
        """
        print(f"Bắt đầu insert {len(documents)} tài liệu vào collection '{collection.name}'...")

        with collection.batch.dynamic() as batch: # type: ignore
            # 1. Hàm sinh batch tài liệu
            def doc_generator(docs: List[Document], bs: int):
                for i in range(0, len(docs), bs):
                    yield docs[i : i + bs]

            # 2. Lặp qua từng batch tài liệu
            for doc_batch in tqdm(doc_generator(documents, batch_size), 
                                  total=(len(documents) + batch_size - 1) // batch_size, 
                                  desc="Embedding & Inserting Batches"):
                
                # 3. Lấy nội dung text của batch
                text_batch = [doc.page_content for doc in doc_batch]
                
                # 4. (# <<< THAY ĐỔI 1: BATCH EMBEDDING)
                # Gọi embed_documents (số nhiều) 1 LẦN cho cả batch
                # Đây là thay đổi mang lại tốc độ nhanh nhất!
                try:
                    vector_batch = self.embed_model.embed_documents(text_batch)
                except Exception as e:
                    print(f"Lỗi khi embedding batch: {e}. Bỏ qua batch này.")
                    continue
                    
                # 5. (# <<< THAY ĐỔI 2: WEAVIATE BATCH MANAGER)
                # Thêm từng object vào trình quản lý batch của Weaviate
                # Weaviate sẽ lo phần còn lại (gửi đi khi đủ batch, đa luồng, v.v.)
                for doc, vector in zip(doc_batch, vector_batch):
                    
                    data_object = {
                        "content": doc.page_content,
                        **doc.metadata  # Thêm tất cả metadata (ví dụ: cord_uid, title...)
                    }
                    
                    batch.add_object(
                        properties=data_object,
                        vector=vector
                    )
        
        # Hàm `with` kết thúc, Weaviate sẽ tự động gửi nốt batch cuối cùng (nếu còn)
        print(f"Data insertion complete.")
        print(f"Tổng số objects trong collection '{collection.name}': {collection.aggregate.over_all(total_count=True).total_count}")