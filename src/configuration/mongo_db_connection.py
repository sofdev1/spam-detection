"""
MongoDB client wrapper. Connection string is read from the
MONGODB_URL environment variable (see .env.example) so credentials
never get hardcoded or committed. Matches the DB/collection names
used in scripts/upload_data_mongodb.py ("pws_projects" / "spam_ham").
"""
import os
import sys
import certifi
import pymongo

from src.constant.training_pipeline import DATABASE_NAME, MONGODB_URL_KEY
from src.exception import SpamDetectionException
from src.logger import logging

ca = certifi.where()


class MongoDBClient:
    client = None

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment variable '{MONGODB_URL_KEY}' is not set")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection established")
        except Exception as e:
            raise SpamDetectionException(e, sys) from e
