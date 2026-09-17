import chromadb
from src_mcp.config import PnLCOBCollection, TraderFundChargesCollection, TraderLimitCollection, FundUSDAumCollection, DB_NAME1
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
                "This collection contains Daily, Monthly and Yearly PnLs (Profit and Losses) of RV Capital firm for various valuation dates - against each of its fund splitted by tradename/asset/themes/traders.",
                "This collection Contains Balance sheet Charges and Expenses done by Trader on that ValuationDate.",
                "This collection contains Capital Limit's that a Trader is allocated in a month",
                "This collection Contains AUM in USD of the Fund for various valuation dates ",
            ],
            metadatas=[
                {"db_type": "MongoDB", "db":DB_NAME1, "target": PnLCOBCollection, "schema": SCHEMA1.format(COLLECTION=PnLCOBCollection)},
                {"db_type": "MongoDB", "db":DB_NAME1, "target": TraderFundChargesCollection, "schema": SCHEMA2.format(COLLECTION=TraderFundChargesCollection)},
                {"db_type": "MongoDB", "db":DB_NAME1, "target": TraderLimitCollection, "schema": SCHEMA3.format(COLLECTION=TraderLimitCollection)},
                {"db_type": "MongoDB", "db":DB_NAME1, "target": FundUSDAumCollection, "schema": SCHEMA4.format(COLLECTION=FundUSDAumCollection)},
            ],
            ids=["id_mongo_1", "id_mongo_2", "id_mongo_3", "id_mongo_4"]
        )
    else:
        logger.info(f"Index already exists...")
        collection = client.get_collection(name=collection_name)

    logger.info(f"Chroma DB client connected")
    return collection