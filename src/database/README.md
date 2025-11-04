# Database Folder

This folder contains all database-related files for the Vehicle Tracking System.

## Files

1. [database_schema.sql](file:///Users/vedprakashchaubey/Desktop/ultimateALPR-sdk/database/database_schema.sql) - SQL schema definition for the vehicle tracking database
2. [vehicle_tracking_config.py](file:///Users/vedprakashchaubey/Desktop/ultimateALPR-sdk/database/vehicle_tracking_config.py) - Configuration settings for the database and tracking system

## Purpose

This folder organizes all database-related components to make the project structure cleaner and more maintainable. The MongoDB implementation is in the main vehicle_tracking_system_mongodb.py file, while the SQL schema and configuration files are organized here.

## Usage

The database configuration and schema files are imported by the main tracking system files. The sys.path is updated in the main Python files to include this directory for proper imports.