#!/usr/bin/env python3
"""
Utility script to show the contents of the MongoDB vehicle tracking database.
"""

import sys
import os
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Add the database directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tracking'))

import database.vehicle_tracking_config as vehicle_tracking_config

# Default MongoDB configuration
DEFAULT_MONGO_URI = "mongodb://localhost:27017/"
DEFAULT_DB_NAME = "vehicle_tracking"

def connect_to_mongodb(mongo_uri=DEFAULT_MONGO_URI, db_name=DEFAULT_DB_NAME):
    """
    Connect to MongoDB database.
    
    Args:
        mongo_uri (str): MongoDB connection URI
        db_name (str): Database name
        
    Returns:
        tuple: (client, db) MongoDB client and database objects
    """
    try:
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000
        )
        db = client[db_name]
        # Test connection
        client.admin.command('ping')
        print(f"✅ Connected to MongoDB at {mongo_uri}/{db_name}")
        return client, db
    except ConnectionFailure as e:
        print(f"❌ MongoDB connection failed: {e}")
        return None, None

def show_collections(db):
    """Show all collections in the database."""
    collections = db.list_collection_names()
    print("\n📋 Available collections:")
    for collection in collections:
        count = db[collection].count_documents({})
        print(f"  - {collection} ({count} documents)")
    print()

def show_collection_data(db, collection_name, limit=10):
    """Show data from a specific collection."""
    try:
        collection = db[collection_name]
        count = collection.count_documents({})
        print(f"\n📊 Collection: {collection_name} ({count} documents)")
        print("=" * 50)
        
        if count == 0:
            print("  No documents found.")
            return
            
        documents = collection.find().sort("_id", -1).limit(limit)
        
        for i, doc in enumerate(documents, 1):
            print(f"\n  Document {i}:")
            for key, value in doc.items():
                if key == "_id":
                    print(f"    {key}: {value}")
                elif isinstance(value, datetime):
                    print(f"    {key}: {value.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print(f"    {key}: {value}")
                    
    except Exception as e:
        print(f"❌ Error retrieving data from {collection_name}: {e}")

def show_database_stats(db):
    """Show database statistics."""
    try:
        print("\n📈 Database Statistics:")
        print("=" * 30)
        
        # Collection counts
        collections = db.list_collection_names()
        for collection_name in collections:
            count = db[collection_name].count_documents({})
            print(f"  {collection_name}: {count} documents")
            
        # Storage info
        db_stats = db.command("dbStats")
        print(f"\n  Data size: {db_stats.get('dataSize', 0) / (1024*1024):.2f} MB")
        print(f"  Storage size: {db_stats.get('storageSize', 0) / (1024*1024):.2f} MB")
        print(f"  Collections: {db_stats.get('collections', 0)}")
        
    except Exception as e:
        print(f"❌ Error retrieving database stats: {e}")

def main():
    """Main function to show database contents."""
    print("🗄️  Vehicle Tracking Database Viewer")
    print("=" * 40)
    
    # Connect to MongoDB
    client, db = connect_to_mongodb()
    if db is None:
        return 1
    
    # Show collections
    show_collections(db)
    
    # Show database stats
    show_database_stats(db)
    
    # Show data from each collection
    collections = db.list_collection_names()
    for collection in collections:
        show_collection_data(db, collection, 5)  # Show first 5 documents
    
    # Close connection
    if client:
        client.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())