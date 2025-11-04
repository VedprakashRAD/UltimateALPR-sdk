#!/usr/bin/env python3
"""
Utility script to automatically clean up old images from the CCTV_photos directory
to manage storage space while preserving important data.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import shutil

# Add the database directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tracking'))

try:
    from database.vehicle_tracking_config import PATHS_CONFIG
    CCTV_PHOTOS_DIR = PATHS_CONFIG["image_storage"]
except ImportError:
    CCTV_PHOTOS_DIR = "./CCTV_photos"

def get_image_files(directory):
    """
    Get all image files in the directory.
    
    Args:
        directory (str): Directory to scan
        
    Returns:
        list: List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    if not os.path.exists(directory):
        print(f"❌ Directory does not exist: {directory}")
        return image_files
        
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
                
    return image_files

def parse_timestamp_from_filename(filename):
    """
    Parse timestamp from filename if it follows the naming convention.
    
    Args:
        filename (str): Filename to parse
        
    Returns:
        datetime: Parsed timestamp or None if not found
    """
    # Try to extract timestamp from filename
    # Expected format: prefix_YYYYMMDD_HHMMSS.extension
    try:
        # Remove extension
        name_without_ext = Path(filename).stem
        
        # Split by underscore and look for timestamp pattern
        parts = name_without_ext.split('_')
        for i in range(len(parts) - 1):
            # Check if we have a date pattern (YYYYMMDD) followed by time pattern (HHMMSS)
            if (len(parts[i]) == 8 and parts[i].isdigit() and 
                len(parts[i+1]) == 6 and parts[i+1].isdigit()):
                timestamp_str = parts[i] + parts[i+1]
                return datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
    except Exception as e:
        # If we can't parse the timestamp, return None
        return None
    
    return None

def delete_old_images(directory, days_old=30, dry_run=False):
    """
    Delete images older than specified days.
    
    Args:
        directory (str): Directory to clean up
        days_old (int): Delete images older than this many days
        dry_run (bool): If True, only show what would be deleted without actually deleting
        
    Returns:
        tuple: (deleted_count, deleted_size_mb)
    """
    cutoff_date = datetime.now() - timedelta(days=days_old)
    deleted_count = 0
    deleted_size = 0
    
    print(f"🔍 Scanning directory: {directory}")
    print(f"📅 Cutoff date: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🧪 Dry run mode: {'ON' if dry_run else 'OFF'}")
    print("-" * 50)
    
    image_files = get_image_files(directory)
    
    if not image_files:
        print("📭 No image files found in directory")
        return 0, 0
    
    print(f"📁 Found {len(image_files)} image files")
    
    for image_path in image_files:
        try:
            # Get file modification time
            mod_time = datetime.fromtimestamp(os.path.getmtime(image_path))
            
            # Try to parse timestamp from filename as fallback
            filename_timestamp = parse_timestamp_from_filename(os.path.basename(image_path))
            if filename_timestamp:
                # Use the older of the two timestamps
                file_time = min(mod_time, filename_timestamp)
            else:
                file_time = mod_time
            
            # Check if file is older than cutoff date
            if file_time < cutoff_date:
                file_size = os.path.getsize(image_path) / (1024 * 1024)  # Size in MB
                
                if not dry_run:
                    os.remove(image_path)
                    print(f"🗑️  Deleted: {os.path.basename(image_path)} ({file_size:.2f} MB) - {file_time.strftime('%Y-%m-%d')}")
                else:
                    print(f"📋 Would delete: {os.path.basename(image_path)} ({file_size:.2f} MB) - {file_time.strftime('%Y-%m-%d')}")
                
                deleted_count += 1
                deleted_size += file_size
                
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
    
    print("-" * 50)
    if not dry_run:
        print(f"✅ Deleted {deleted_count} files ({deleted_size:.2f} MB)")
    else:
        print(f"📋 Would delete {deleted_count} files ({deleted_size:.2f} MB)")
    
    return deleted_count, deleted_size

def get_directory_size(directory):
    """
    Calculate the total size of a directory.
    
    Args:
        directory (str): Directory to calculate size for
        
    Returns:
        float: Size in MB
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"❌ Error calculating directory size: {e}")
    
    return total_size / (1024 * 1024)  # Convert to MB

def show_storage_info(directory):
    """
    Show storage information for the directory.
    
    Args:
        directory (str): Directory to show info for
    """
    if not os.path.exists(directory):
        print(f"❌ Directory does not exist: {directory}")
        return
    
    total_size = get_directory_size(directory)
    file_count = len(get_image_files(directory))
    
    print(f"📊 Storage Information for {directory}:")
    print(f"   Total files: {file_count}")
    print(f"   Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    print()

def main():
    """Main function to handle command line arguments."""
    print("🧹 CCTV Photos Cleanup Utility")
    print("=" * 40)
    
    # Show current storage info
    show_storage_info(CCTV_PHOTOS_DIR)
    
    # If no arguments provided, show help
    if len(sys.argv) < 2:
        print("📋 Usage:")
        print("  python cleanup_old_images.py info                    - Show storage information")
        print("  python cleanup_old_images.py clean [days] [--dry-run] - Clean images older than days")
        print("  python cleanup_old_images.py clean 30                - Delete images older than 30 days")
        print("  python cleanup_old_images.py clean 7 --dry-run       - Preview deletion of images older than 7 days")
        print()
        return 0
    
    command = sys.argv[1].lower()
    
    if command == "info":
        return 0
    
    elif command == "clean":
        if len(sys.argv) < 3:
            print("❌ Please provide number of days")
            return 1
            
        try:
            days = int(sys.argv[2])
            dry_run = "--dry-run" in sys.argv
            
            # Show what would be deleted (dry run)
            deleted_count, deleted_size = delete_old_images(CCTV_PHOTOS_DIR, days, dry_run=True)
            
            if not dry_run and deleted_count > 0:
                print()
                confirm = input(f"⚠️  Confirm deletion of {deleted_count} files ({deleted_size:.2f} MB)? (y/N): ")
                if confirm.lower() in ['y', 'yes']:
                    # Actually delete files
                    delete_old_images(CCTV_PHOTOS_DIR, days, dry_run=False)
                else:
                    print("❌ Deletion cancelled")
        except ValueError:
            print("❌ Please provide a valid number of days")
            return 1
        except KeyboardInterrupt:
            print("\n🛑 Operation cancelled by user")
            return 1
            
    else:
        print(f"❌ Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
