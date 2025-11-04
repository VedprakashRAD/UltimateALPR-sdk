#!/usr/bin/env python3
"""
Utility script to delete entries from the MongoDB vehicle tracking database.
"""

import sys
import os
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Add the database directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tracking'))

# Fixed import - use relative import
try:
    from database.vehicle_tracking_config import DATABASE_PATH
except ImportError:
    # Fallback to direct path
    DATABASE_PATH = "vehicle_tracking.db"

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

def list_collections(db):
    """List all collections in the database."""
    collections = db.list_collection_names()
    print("\n📋 Available collections:")
    for collection in collections:
        count = db[collection].count_documents({})
        print(f"  - {collection} ({count} documents)")
    print()

def delete_entry_by_id(db, entry_id):
    """
    Delete a specific entry by its ID.
    
    Args:
        db: MongoDB database object
        entry_id (str): Entry ID to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        from bson import ObjectId
        result = db.entry_events.delete_one({"_id": ObjectId(entry_id)})
        if result.deleted_count > 0:
            print(f"✅ Deleted entry with ID: {entry_id}")
            return True
        else:
            print(f"❌ No entry found with ID: {entry_id}")
            return False
    except Exception as e:
        print(f"❌ Error deleting entry: {e}")
        return False

def delete_entries_by_plate_number(db, plate_number):
    """
    Delete all entries with a specific plate number.
    
    Args:
        db: MongoDB database object
        plate_number (str): Plate number to delete entries for
        
    Returns:
        int: Number of deleted entries
    """
    try:
        # Delete from entry_events
        result1 = db.entry_events.delete_many({
            "$or": [
                {"front_plate_number": plate_number},
                {"rear_plate_number": plate_number}
            ]
        })
        
        # Delete from exit_events
        result2 = db.exit_events.delete_many({
            "$or": [
                {"front_plate_number": plate_number},
                {"rear_plate_number": plate_number}
            ]
        })
        
        # Delete from vehicle_journeys
        result3 = db.vehicle_journeys.delete_many({
            "$or": [
                {"front_plate_number": plate_number},
                {"rear_plate_number": plate_number}
            ]
        })
        
        total_deleted = result1.deleted_count + result2.deleted_count + result3.deleted_count
        print(f"✅ Deleted {result1.deleted_count} entry events")
        print(f"✅ Deleted {result2.deleted_count} exit events")
        print(f"✅ Deleted {result3.deleted_count} vehicle journeys")
        print(f"✅ Total deleted: {total_deleted} documents")
        
        return total_deleted
    except Exception as e:
        print(f"❌ Error deleting entries by plate number: {e}")
        return 0

def delete_entries_older_than(db, days):
    """
    Delete entries older than a specified number of days.
    
    Args:
        db: MongoDB database object
        days (int): Number of days
        
    Returns:
        int: Number of deleted entries
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Delete from entry_events
        result1 = db.entry_events.delete_many({
            "entry_timestamp": {"$lt": cutoff_date}
        })
        
        # Delete from exit_events
        result2 = db.exit_events.delete_many({
            "exit_timestamp": {"$lt": cutoff_date}
        })
        
        # Delete from vehicle_journeys
        result3 = db.vehicle_journeys.delete_many({
            "entry_timestamp": {"$lt": cutoff_date}
        })
        
        total_deleted = result1.deleted_count + result2.deleted_count + result3.deleted_count
        print(f"✅ Deleted {result1.deleted_count} entry events older than {days} days")
        print(f"✅ Deleted {result2.deleted_count} exit events older than {days} days")
        print(f"✅ Deleted {result3.deleted_count} vehicle journeys older than {days} days")
        print(f"✅ Total deleted: {total_deleted} documents")
        
        return total_deleted
    except Exception as e:
        print(f"❌ Error deleting old entries: {e}")
        return 0

def delete_all_entries(db):
    """
    Delete all entries from all collections.
    
    Args:
        db: MongoDB database object
        
    Returns:
        dict: Deletion results for each collection
    """
    try:
        # Count documents before deletion
        entry_count = db.entry_events.count_documents({})
        exit_count = db.exit_events.count_documents({})
        journey_count = db.vehicle_journeys.count_documents({})
        employee_count = db.employee_vehicles.count_documents({})
        
        print(f"⚠️  This will delete ALL data from the database!")
        print(f"  - {entry_count} entry events")
        print(f"  - {exit_count} exit events")
        print(f"  - {journey_count} vehicle journeys")
        print(f"  - {employee_count} employee vehicles")
        
        confirm = input("\nType 'DELETE ALL' to confirm: ")
        if confirm != "DELETE ALL":
            print("❌ Deletion cancelled")
            return {}
        
        # Delete all documents
        result1 = db.entry_events.delete_many({})
        result2 = db.exit_events.delete_many({})
        result3 = db.vehicle_journeys.delete_many({})
        result4 = db.employee_vehicles.delete_many({})
        
        results = {
            "entry_events": result1.deleted_count,
            "exit_events": result2.deleted_count,
            "vehicle_journeys": result3.deleted_count,
            "employee_vehicles": result4.deleted_count
        }
        
        print(f"✅ Deleted {result1.deleted_count} entry events")
        print(f"✅ Deleted {result2.deleted_count} exit events")
        print(f"✅ Deleted {result3.deleted_count} vehicle journeys")
        print(f"✅ Deleted {result4.deleted_count} employee vehicles")
        
        return results
    except Exception as e:
        print(f"❌ Error deleting all entries: {e}")
        return {}

def show_entry_details(db, entry_id):
    """
    Show details of a specific entry.
    
    Args:
        db: MongoDB database object
        entry_id (str): Entry ID to show details for
    """
    try:
        from bson import ObjectId
        entry = db.entry_events.find_one({"_id": ObjectId(entry_id)})
        if entry:
            print(f"\n📋 Entry Details for ID: {entry_id}")
            print(f"  Front Plate: {entry.get('front_plate_number', 'N/A')}")
            print(f"  Rear Plate: {entry.get('rear_plate_number', 'N/A')}")
            print(f"  Entry Time: {entry.get('entry_timestamp', 'N/A')}")
            print(f"  Processed: {entry.get('is_processed', False)}")
            print(f"  Front Image: {entry.get('front_plate_image_path', 'N/A')}")
            print(f"  Rear Image: {entry.get('rear_plate_image_path', 'N/A')}")
        else:
            print(f"❌ No entry found with ID: {entry_id}")
    except Exception as e:
        print(f"❌ Error retrieving entry details: {e}")

def show_recent_entries(db, limit=10):
    """
    Show recent entries.
    
    Args:
        db: MongoDB database object
        limit (int): Number of recent entries to show
    """
    try:
        print(f"\n📋 {limit} Most Recent Entries:")
        entries = db.entry_events.find().sort("entry_timestamp", -1).limit(limit)
        
        for i, entry in enumerate(entries, 1):
            print(f"  {i}. ID: {entry['_id']}")
            print(f"     Plate: {entry.get('front_plate_number', 'N/A')}")
            print(f"     Time: {entry.get('entry_timestamp', 'N/A')}")
            print(f"     Processed: {entry.get('is_processed', False)}")
            print()
    except Exception as e:
        print(f"❌ Error retrieving recent entries: {e}")

def main():
    """Main function to handle command line arguments."""
    print("🗑️  Vehicle Tracking Database Entry Deletion Tool")
    print("=" * 50)
    
    # Connect to MongoDB
    client, db = connect_to_mongodb()
    if db is None:
        return 1
    
    # List available collections
    list_collections(db)
    
    # If no arguments provided, show help
    if len(sys.argv) < 2:
        print("📋 Usage:")
        print("  python delete_entries.py list [limit]     - List recent entries")
        print("  python delete_entries.py show <id>        - Show entry details")
        print("  python delete_entries.py delete <id>      - Delete entry by ID")
        print("  python delete_entries.py plate <number>   - Delete entries by plate number")
        print("  python delete_entries.py old <days>       - Delete entries older than N days")
        print("  python delete_entries.py all              - Delete ALL entries (⚠️  dangerous)")
        print()
        return 0
    
    command = sys.argv[1].lower()
    
    if command == "list":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        show_recent_entries(db, limit)
        
    elif command == "show":
        if len(sys.argv) < 3:
            print("❌ Please provide an entry ID")
            return 1
        entry_id = sys.argv[2]
        show_entry_details(db, entry_id)
        
    elif command == "delete":
        if len(sys.argv) < 3:
            print("❌ Please provide an entry ID")
            return 1
        entry_id = sys.argv[2]
        delete_entry_by_id(db, entry_id)
        
    elif command == "plate":
        if len(sys.argv) < 3:
            print("❌ Please provide a plate number")
            return 1
        plate_number = sys.argv[2]
        delete_entries_by_plate_number(db, plate_number)
        
    elif command == "old":
        if len(sys.argv) < 3:
            print("❌ Please provide number of days")
            return 1
        try:
            days = int(sys.argv[2])
            delete_entries_older_than(db, days)
        except ValueError:
            print("❌ Please provide a valid number of days")
            return 1
            
    elif command == "all":
        delete_all_entries(db)
        
    else:
        print(f"❌ Unknown command: {command}")
        return 1
    
    # Close connection
    if client:
        client.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())