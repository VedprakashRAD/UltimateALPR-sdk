#!/usr/bin/env python3
"""
Indian License Plate Validator and Analyzer
Supports all Indian number plate formats and types
"""

import re
from datetime import datetime

class IndianPlateValidator:
    def __init__(self):
        # State and UT codes
        self.states = {
            'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh', 'AS': 'Assam', 'BR': 'Bihar',
            'CG': 'Chhattisgarh', 'GA': 'Goa', 'GJ': 'Gujarat', 'HR': 'Haryana',
            'HP': 'Himachal Pradesh', 'JH': 'Jharkhand', 'KA': 'Karnataka', 'KL': 'Kerala',
            'MP': 'Madhya Pradesh', 'MH': 'Maharashtra', 'MN': 'Manipur', 'ML': 'Meghalaya',
            'MZ': 'Mizoram', 'NL': 'Nagaland', 'OD': 'Odisha', 'PB': 'Punjab',
            'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TS': 'Telangana',
            'TR': 'Tripura', 'UP': 'Uttar Pradesh', 'UK': 'Uttarakhand', 'WB': 'West Bengal'
        }
        
        self.union_territories = {
            'AN': 'Andaman and Nicobar', 'CH': 'Chandigarh', 'DN': 'Dadra and Nagar Haveli',
            'DD': 'Daman and Diu', 'DL': 'Delhi', 'JK': 'Jammu and Kashmir',
            'LA': 'Ladakh', 'LD': 'Lakshadweep', 'PY': 'Puducherry'
        }
        
        # Plate patterns
        self.patterns = {
            'standard': r'^([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})$',
            'bharat': r'^(\d{2})BH(\d{4})([A-Z]{2})$',
            'vintage': r'^([A-Z]{2})VA([A-Z]{2})(\d{4})$',
            'temporary': r'^T(\d{4})([A-Z]{2})(\d{4})([A-Z]{2})$',
            'trade': r'^([A-Z]{2})(\d{2})([A-Z])(\d{4})TC(\d{4})$',
            'military': r'^↑(\d{2})([A-Z])(\d{6})([A-Z])$'
        }
    
    def validate_plate(self, plate_text):
        """Validate and analyze Indian license plate."""
        if not plate_text or len(plate_text) < 6:
            return {'valid': False, 'reason': 'Too short'}
        
        plate = plate_text.upper().replace(' ', '')
        
        # Check each pattern
        for plate_type, pattern in self.patterns.items():
            match = re.match(pattern, plate)
            if match:
                return self.analyze_plate(plate, plate_type, match)
        
        return {'valid': False, 'reason': 'Invalid format', 'plate': plate}
    
    def analyze_plate(self, plate, plate_type, match):
        """Analyze matched plate and extract information."""
        result = {
            'valid': True,
            'plate': plate,
            'type': plate_type,
            'format': self.get_format_description(plate_type)
        }
        
        if plate_type == 'standard':
            state_code, rto_code, series, number = match.groups()
            result.update({
                'state_code': state_code,
                'state_name': self.get_state_name(state_code),
                'rto_code': rto_code,
                'series': series,
                'number': number,
                'vehicle_category': self.get_vehicle_category(series)
            })
            
        elif plate_type == 'bharat':
            year, unique_id, code = match.groups()
            result.update({
                'registration_year': f"20{year}",
                'unique_id': unique_id,
                'code': code,
                'pan_india': True,
                'benefits': ['No re-registration needed', 'Valid across India']
            })
            
        elif plate_type == 'vintage':
            state_code, series, number = match.groups()
            result.update({
                'state_code': state_code,
                'state_name': self.get_state_name(state_code),
                'series': series,
                'number': number,
                'age': '50+ years',
                'restrictions': ['Exhibition use only', 'Rally participation']
            })
        
        return result
    
    def get_state_name(self, code):
        """Get full state name from code."""
        return self.states.get(code) or self.union_territories.get(code) or 'Unknown'
    
    def get_format_description(self, plate_type):
        """Get format description."""
        descriptions = {
            'standard': 'Standard Indian format (XX00XX0000)',
            'bharat': 'Bharat Series (00BH0000XX)',
            'vintage': 'Vintage Vehicle (XXVA000000)',
            'temporary': 'Temporary Registration',
            'trade': 'Trade Certificate',
            'military': 'Military Vehicle'
        }
        return descriptions.get(plate_type, 'Unknown format')
    
    def get_vehicle_category(self, series):
        """Determine vehicle category from series."""
        if len(series) == 1:
            return 'Two-wheeler'
        elif len(series) == 2:
            return 'Four-wheeler'
        else:
            return 'Commercial/Special'
    
    def get_plate_color_info(self, plate_info):
        """Determine expected plate color based on type."""
        if plate_info.get('type') == 'bharat':
            return {'background': 'White/Yellow', 'text': 'Black'}
        elif plate_info.get('vehicle_category') == 'Commercial/Special':
            return {'background': 'Yellow', 'text': 'Black'}
        else:
            return {'background': 'White', 'text': 'Black'}

# Global validator instance
validator = IndianPlateValidator()

def validate_indian_plate(plate_text):
    """Main validation function."""
    return validator.validate_plate(plate_text)