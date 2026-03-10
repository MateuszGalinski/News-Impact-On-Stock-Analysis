from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, WeightedRanker
from sentence_transformers import SentenceTransformer
from typing import Union, List, Dict, Any
from data_classes import NewsItem
from dotenv import load_dotenv
import os

load_dotenv()
    

class BGEEmbedding:
    MODEL = "BAAI/bge-m3"

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL)

    @property
    def vector_size(self) -> int | None:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str):
        output = self.model.encode(text, output_value='sentence_embedding')
        return output 


class NewsVectorDatabase:
    embeddings = BGEEmbedding()
    VECTOR_SIZE = embeddings.vector_size
    DATABASE_NAME = "StockNews"
    COLLECTIONS = {
        "news": {
            "vector_field": "text_dense",
            "dimension": VECTOR_SIZE,
            "fields": [
                {"field_name": "id",        "datatype": DataType.INT64,   "is_primary": True, "auto_id": True},
                {"field_name": "created_at", "datatype": DataType.INT64},
                {"field_name": "symbols",   "datatype": DataType.VARCHAR, "max_length": 1000},
                {"field_name": "headline", "datatype": DataType.VARCHAR, "max_length": 1000},
                # Hybrid search fields
                {"field_name": "text_raw",    "datatype": DataType.VARCHAR, "max_length": 20000, "enable_analyzer": True},
                {"field_name": "text_sparse", "datatype": DataType.SPARSE_FLOAT_VECTOR},
                {"field_name": "text_dense",  "datatype": DataType.FLOAT_VECTOR, "dim": VECTOR_SIZE},
            ],
            "functions": [
                Function(
                    name="bm25_text",
                    input_field_names=["text_raw"],
                    output_field_names=["text_sparse"],
                    function_type=FunctionType.BM25,
                )
            ],
        }
    }

    def __init__(self, uri = None, token = None):
        # 'ENGINE': 'milvus',
        # 'TOKEN': "root:Milvus",
        # 'SOURCE': os.environ.get('MILVUS_DB_SOURCE')
        if not uri:
            uri = os.environ.get('MILVUS_DB_SOURCE')

        if not token:
            token = os.environ.get('MILVUS_DB_TOKEN')
        
        if uri and token:
            self.client = MilvusClient(uri = uri, token = token)
        else:
            raise Exception('Milvus engine source or token missing') 

        self._ensure_database()

        for name, config in self.COLLECTIONS.items():
            self._ensure_collection(name, config)
    
    def _ensure_collection(self, name, config):
        if not self.client.has_collection(name):
            schema = self._create_schema(config['fields'], config.get('functions'))
            self.client.create_collection(
                collection_name=name,
                dimension=config['dimension'],
                schema=schema
            )
            self.client.create_index(
                collection_name=name,
                index_params=self._create_index_params(config['vector_field'])
            )
            self.client.create_index(
                collection_name=name,
                index_params=self._create_sparse_index_params(config['vector_field'].replace("_dense", "_sparse"))
            )

        self.client.load_collection(name)
            
    def _ensure_database(self):
        """Ensures the database exists and switches to it."""
        existing_dbs = self.client.list_databases()
        if self.DATABASE_NAME not in existing_dbs:
            self.client.create_database(self.DATABASE_NAME)
        self.client.use_database(
            db_name=self.DATABASE_NAME
        )

    def _create_schema(self, fields, functions=None):
        schema = MilvusClient.create_schema()
        for field in fields:
            schema.add_field(**field)
        if functions:
            for function in functions:
                schema.add_function(function)
        return schema

    def _create_index_params(self, vector_field_name):
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=vector_field_name,
            index_type="FLAT",
            index_name=f"{vector_field_name}_index",
            metric_type="COSINE",
            params={"nlist": 64}
        )
        return index_params
    
    def _create_sparse_index_params(self, sparse_field_name):
        index_params = MilvusClient.prepare_index_params()
        sparse_field = sparse_field_name
        index_params.add_index(
            field_name=sparse_field,
            index_name=f"{sparse_field}_sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_WAND"}, # or "DAAT_WAND" or "TAAT_NAIVE"
        )
        return index_params
    
    def add_news(self, news: Union[Dict[str, Any], NewsItem, List[Union[Dict[str, Any], NewsItem]]]) -> None:
        """
        Insert one or more news articles into the 'news' collection.
        
        Args:
            news: A single news item (dict or NewsItem) or a list of them.
        """
        # Normalize input to a list of NewsItem objects
        if not isinstance(news, list):
            news = [news]

        items : List[NewsItem] = []
        for entry in news:
            if isinstance(entry, dict):
                # Convert dict to NewsItem (ensures required fields exist)
                items.append(NewsItem(**entry))
            elif isinstance(entry, NewsItem):
                items.append(entry)
            else:
                raise TypeError("Each news item must be a dict or NewsItem")

        # Prepare data for insertion
        data_to_insert = []
        for item in items:
            combined : str = item.combined_text()
            embedding = self.embeddings.embed(combined)  # dense vector

            data_to_insert.append({
                "created_at": item.to_timestamp(),
                "symbols": item.symbols,
                "headline": item.headline,
                "text_raw": combined,                     # used by BM25
                "text_dense": embedding.tolist(),          # dense vector
                # id is auto-generated, so not included
            })

        # Insert into Milvus
        self.client.insert(collection_name="news", data=data_to_insert)

    def drop_database(self) -> None:
        """
        Drops all collections and the database itself.
        """
        for name in self.COLLECTIONS.keys():
            if self.client.has_collection(name):
                self.client.release_collection(name)
                self.client.drop_collection(name)

        existing_dbs = self.client.list_databases()
        if self.DATABASE_NAME in existing_dbs:
            self.client.drop_database(self.DATABASE_NAME)

    def search(
        self,
        text: str,
        top_k: int = 5,
        output_fields: List[str] | None = None,
        filter: str = "",
        collection_name: str = "news",
    ) -> List[Dict[str, Any]]:
        """
        Dense vector search against the collection.

        Args:
            text: Natural language query.
            top_k: Number of results to return.
            output_fields: Fields to include in results. Defaults to all fields.
            filter: Optional Milvus filter expression e.g. 'symbols like "%AAPL%"'.
            collection_name: Collection to search in.
        """
        collection_info = self.COLLECTIONS.get(collection_name)
        if not collection_info:
            raise ValueError(f"Invalid collection name: {collection_name}")

        if output_fields is None:
            output_fields = ["*"]

        dense_vector = self.embeddings.embed(text).tolist()

        raw_results = self.client.search(
            collection_name=collection_name,
            data=[dense_vector],
            anns_field=collection_info["vector_field"],
            limit=top_k,
            output_fields=output_fields,
            filter=filter,
        )

        return [
            {"id": hit["id"], "distance": hit["distance"], **hit["entity"]}
            for hits in raw_results
            for hit in hits
        ]

    def search_hybrid(
        self,
        text: str,
        top_k: int = 6,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        output_fields: List[str] | None = None,
        filter: str = "",
        collection_name: str = "news",
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining dense (COSINE) and sparse (BM25) vectors via WeightedRanker.

        Args:
            text: Natural language query.
            top_k: Number of results to return.
            dense_weight: Weight for dense vector score (0-1).
            sparse_weight: Weight for sparse/BM25 score (0-1).
            output_fields: Fields to include in results. Defaults to all fields.
            filter: Optional Milvus filter expression e.g. 'symbols like "%AAPL%"'.
            collection_name: Collection to search in.
        """
        collection_info = self.COLLECTIONS.get(collection_name)
        if not collection_info:
            raise ValueError(f"Invalid collection name: {collection_name}")

        if output_fields is None:
            output_fields = ["*"]

        dense_field = collection_info["vector_field"]
        sparse_field = dense_field.replace("_dense", "_sparse")

        dense_vector = self.embeddings.embed(text).tolist()

        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field=dense_field,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=filter,
        )

        sparse_req = AnnSearchRequest(
            data=[text],
            anns_field=sparse_field,
            param={"metric_type": "BM25", "params": {}},
            limit=top_k,
            expr=filter,
        )

        ranker = WeightedRanker(dense_weight, sparse_weight)

        raw_results = self.client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=top_k,
            output_fields=output_fields,
            filter=filter,
        )

        return [
            {"id": hit["id"], "distance": hit["distance"], **hit["entity"]}
            for hit in raw_results[0]
        ]

# __________________________________________TESTING___________________________________________________

def main():
    # Initialize DB (will create database and collection if not exist)
    db = NewsVectorDatabase(
        uri=os.getenv("MILVUS_DB_SOURCE", "http://localhost:19530"),
        token=os.getenv("MILVUS_DB_TOKEN", "root:Milvus")
    )

    # Sample news items (as dicts)
    news1 = {
        "created_at": "2026-02-26 08:49:31+00:00",
        "symbols": "AMZN,MSFT",
        "headline": "Amazon's $50 Billion Investment In OpenAI Could Hinge On IPO, AGI",
        "summary": "",
        "full_text": "Amazon's $50 Billion Investment In OpenAI Could Hinge On IPO, AGI - The Information.",
        "source": "benzinga",
        "url": "https://www.benzinga.com/news/26/02/50878006/amazons-50-billion-investment-in-openai-could-hinge-on-ipo-agi-the-information"
    }

    news2 = {
        "created_at": "2026-02-26 08:00:00+00:00",
        "symbols": "AMZN,GOOG,META,MSFT,TSLA",
        "headline": "Bernie Sanders Targets Elon Musk, Jeff Bezos, Mark Zuckerberg In Senate Speech",
        "summary": "Sen. Bernie Sanders warned that AI could eliminate up to 100 million U.S. jobs and deepen inequality, targeting tech leaders.",
        "full_text": "Bernie Sanders Targets Elon Musk, Jeff Bezos, Mark Zuckerberg In Senate Speech, Proposes Ban On New AI Data Centers: 'Long Overdue'.",
        "source": "benzinga",
        "url": "https://www.benzinga.com/markets/tech/26/02/50877665/bernie-sanders-targets-elon-musk-jeff-bezos-mark-zuckerberg-in-senate-speech-proposes-ban-on-new-ai-data-centers-long-overdue"
    }

    # Add as a list
    db.add_news([news1, news2])

    # Also test with a NewsItem object
    item = NewsItem(
        created_at="2026-02-27 10:00:00+00:00",
        symbols="AAPL",
        headline="Apple announces new AI features",
        summary="Apple unveils on-device AI models",
        full_text="Apple today announced a suite of AI features for iPhone...",
        source="techcrunch",
        url="https://techcrunch.com/..."
    )
    db.add_news(item)

    print("News added successfully!")

    # Optional: perform a search to verify
    query = "AI investment by tech giants"
    query_embedding = db.embeddings.embed(query).tolist()

    results = db.client.search(
        collection_name="news",
        data=[query_embedding],
        anns_field="text_dense",
        limit=5,
        output_fields=["symbols", "headline", "created_at"]
    )

    print("\nSearch results:")
    for hits in results:
        for hit in hits:
            print(f"ID: {hit['id']}, distance: {hit['distance']}")
            print(f"  symbols: {hit['entity']['symbols']}")
            print(f"  headline: {hit['entity']['headline'][:60]}...")
            print()

    db.drop_database()

if __name__ == "__main__":
    main()