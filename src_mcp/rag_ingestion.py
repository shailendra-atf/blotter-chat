import chromadb
from src_mcp.config import COLLECTION1, COLLECTION2, COLLECTION3, COLLECTION4, DB_NAME1
from src_mcp.context import SCHEMA1, SCHEMA2, SCHEMA3, SCHEMA4
from src_mcp.dependencies import logger

# Initialize Local Vector Database containing all metadata descriptions
def initialize_chromadb():
    client = chromadb.PersistentClient(path="chroma_db")
    collection_name = "schemas"
    existing_collections = [c.name for c in client.list_collections()]
    logger.info(f"{existing_collections=}")
    if collection_name not in existing_collections:
        logger.info(f"Index '{collection_name}' not found. Creating and ingesting...")
        
        # Create the new collection safely
        collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        
        # Perform your ingestion pipeline here
        collection.upsert(
            documents=[
                "This collection contains NAVinUSD, Daily, Monthly and Yearly PnL (Profit and Losses) of RV Capital firm for various valuation dates - for each fund / assets / themes / traders / tradenames.",
                # "Contains Balance sheet Charges and Expenses done by Trader on that ValuationDate. This Collection doesn not contain PnL.",
                "This collection contains Authorised Limit's that a Trader can spend.",
                # "Contains AUM in million USD about the Fund",
            ],
            metadatas=[
                {"db_type": "MongoDB", "db":DB_NAME1, "target": COLLECTION1, "schema": SCHEMA1.format(COLLECTION=COLLECTION1)},
                # {"db_type": "MongoDB", "db":DB_NAME1, "target": COLLECTION2, "schema": SCHEMA2.format(COLLECTION=COLLECTION2)},
                {"db_type": "MongoDB", "db":DB_NAME1, "target": COLLECTION3, "schema": SCHEMA3.format(COLLECTION=COLLECTION3)},
                # {"db_type": "MongoDB", "db":DB_NAME1, "target": COLLECTION4, "schema": SCHEMA4.format(COLLECTIO4=COLLECTION4)},
            ],
            ids=["id_mongo_1", "id_mongo_2"]#, "id_mongo_3", "id_mongo_4"]
        )
    else:
        logger.info(f"Index already exists...")
        collection = client.get_collection(name=collection_name)

    logger.info(f"Chroma DB client connected")
    return collection