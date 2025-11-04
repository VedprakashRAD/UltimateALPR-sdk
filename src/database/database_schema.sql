-- Database schema for vehicle entry/exit tracking system

-- Table for storing vehicle information
CREATE TABLE vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL,
    vehicle_type TEXT, -- car, truck, motorcycle, etc.
    vehicle_color TEXT,
    vehicle_make TEXT,
    vehicle_model TEXT,
    is_employee BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing entry events
CREATE TABLE entry_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    front_plate_number TEXT,
    rear_plate_number TEXT,
    front_plate_image_path TEXT,
    rear_plate_image_path TEXT,
    front_plate_confidence REAL,
    rear_plate_confidence REAL,
    entry_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    vehicle_color TEXT,
    vehicle_make TEXT,
    vehicle_model TEXT,
    is_processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing exit events
CREATE TABLE exit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    front_plate_number TEXT,
    rear_plate_number TEXT,
    front_plate_image_path TEXT,
    rear_plate_image_path TEXT,
    front_plate_confidence REAL,
    rear_plate_confidence REAL,
    exit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    vehicle_color TEXT,
    vehicle_make TEXT,
    vehicle_model TEXT,
    is_processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing complete vehicle journeys (matched entry/exit events)
CREATE TABLE vehicle_journeys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_event_id INTEGER,
    exit_event_id INTEGER,
    front_plate_number TEXT,
    rear_plate_number TEXT,
    entry_timestamp TIMESTAMP,
    exit_timestamp TIMESTAMP,
    duration_seconds INTEGER,
    vehicle_color TEXT,
    vehicle_make TEXT,
    vehicle_model TEXT,
    is_employee BOOLEAN DEFAULT FALSE,
    flagged_for_review BOOLEAN DEFAULT FALSE,
    review_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (entry_event_id) REFERENCES entry_events(id),
    FOREIGN KEY (exit_event_id) REFERENCES exit_events(id)
);

-- Table for storing anomalies that need manual review
CREATE TABLE anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT, -- 'entry' or 'exit'
    event_id INTEGER, -- reference to entry_events or exit_events
    anomaly_type TEXT, -- 'mismatched_plates', 'missing_data', 'low_confidence', etc.
    description TEXT,
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better query performance
CREATE INDEX idx_vehicles_plate_number ON vehicles(plate_number);
CREATE INDEX idx_entry_events_timestamp ON entry_events(entry_timestamp);
CREATE INDEX idx_exit_events_timestamp ON exit_events(exit_timestamp);
CREATE INDEX idx_vehicle_journeys_entry_timestamp ON vehicle_journeys(entry_timestamp);
CREATE INDEX idx_vehicle_journeys_exit_timestamp ON vehicle_journeys(exit_timestamp);
CREATE INDEX idx_vehicle_journeys_plate_numbers ON vehicle_journeys(front_plate_number, rear_plate_number);