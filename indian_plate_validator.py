#!/usr/bin/env python3
"""
Indian License Plate Validator and Analyzer
Supports all Indian number plate formats and types
"""

import re
from datetime import datetime

class IndianPlateValidator:
    def __init__(self):
        # All Indian states and UTs
        self.all_codes = {
            'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh', 'AS': 'Assam', 'BR': 'Bihar',
            'CG': 'Chhattisgarh', 'GA': 'Goa', 'GJ': 'Gujarat', 'HR': 'Haryana',
            'HP': 'Himachal Pradesh', 'JH': 'Jharkhand', 'KA': 'Karnataka', 'KL': 'Kerala',
            'MP': 'Madhya Pradesh', 'MH': 'Maharashtra', 'MN': 'Manipur', 'ML': 'Meghalaya',
            'MZ': 'Mizoram', 'NL': 'Nagaland', 'OD': 'Odisha', 'PB': 'Punjab',
            'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TS': 'Telangana',
            'TR': 'Tripura', 'UP': 'Uttar Pradesh', 'UK': 'Uttarakhand', 'WB': 'West Bengal',
            'AN': 'Andaman and Nicobar', 'CH': 'Chandigarh', 'DN': 'Dadra and Nagar Haveli',
            'DD': 'Daman and Diu', 'DL': 'Delhi', 'JK': 'Jammu and Kashmir',
            'LA': 'Ladakh', 'LD': 'Lakshadweep', 'PY': 'Puducherry'
        }
        
        # Comprehensive plate patterns
        self.patterns = {
            'standard': r'^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$',  # XX00XXX0000
            'bharat': r'^(\d{2})BH(\d{4})([A-Z]{2})$',              # 00BH0000XX
            'vintage': r'^([A-Z]{2})VA([A-Z]{2})(\d{4})$',          # XXVA000000
            'temporary': r'^T(\d{4})([A-Z]{2})(\d{3,4})([A-Z]{2})$', # T0000XX000XX
            'trade': r'^([A-Z]{2})(\d{2})([A-Z])(\d{4})TC(\d{4})$', # XX00X0000TC0000
            'military': r'^(\d{2})([A-Z])(\d{6})([A-Z])$',          # 00X000000X
            'diplomatic': r'^([A-Z]{2,3})(\d{3,4})$'                # CC/CD/UN0000
        }
    
    def validate_plate(self, plate_text):
        """Validate and analyze Indian license plate."""
        if not plate_text or len(plate_text) < 6:
            return {'valid': False, 'reason': 'Too short'}
        
        plate = plate_text.upper().replace(' ', '').replace('-', '')
        
        # Check each pattern in order of priority
        for plate_type, pattern in self.patterns.items():
            match = re.match(pattern, plate)
            if match:
                result = self.analyze_plate(plate, plate_type, match)
                if result['valid']:
                    return result
        
        # If no exact match, try strict validation for Indian plates only
        if self.is_likely_indian_plate(plate):
            state_code = plate[:2]
            return {
                'valid': True,
                'plate': plate,
                'type': 'standard',
                'format': 'Standard Indian format',
                'state_name': self.all_codes.get(state_code, 'Unknown'),
                'state_code': state_code,
                'confidence': 'high'
            }
        
        return {'valid': False, 'reason': 'Invalid format', 'plate': plate}
    
    def analyze_plate(self, plate, plate_type, match):
        """Analyze matched plate and extract information."""
        result = {
            'valid': True,
            'plate': plate,
            'type': plate_type,
            'format': self.get_format_description(plate_type),
            'confidence': 'high'
        }
        
        if plate_type == 'standard':
            state_code, rto_code, series, number = match.groups()
            # Validate state code exists
            if state_code not in self.all_codes:
                result['valid'] = False
                return result
            
            result.update({
                'state_code': state_code,
                'state_name': self.all_codes[state_code],
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
                'state_name': 'Bharat Series (Pan-India)',
                'pan_india': True
            })
            
        elif plate_type == 'vintage':
            state_code, series, number = match.groups()
            if state_code not in self.all_codes:
                result['valid'] = False
                return result
            
            result.update({
                'state_code': state_code,
                'state_name': self.all_codes[state_code],
                'series': series,
                'number': number,
                'age': '50+ years'
            })
            
        elif plate_type == 'diplomatic':
            result.update({
                'state_name': 'Diplomatic Mission',
                'special_category': True
            })
        
        return result
    
    def is_likely_indian_plate(self, plate):
        """Check if plate looks like Indian format with strict validation."""
        # Must be proper length
        if len(plate) < 8 or len(plate) > 10:
            return False
        
        # Must start with valid state code
        state_code = plate[:2]
        if state_code not in self.all_codes:
            return False
        
        # Must follow basic Indian pattern: XX##XX####
        if not re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', plate):
            return False
        
        # Additional checks for common invalid patterns
        # Reject plates with too many consecutive same characters
        for i in range(len(plate) - 2):
            if plate[i] == plate[i+1] == plate[i+2]:
                return False
        
        return True
    
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
    """Main validation function with enhanced logic."""
    result = validator.validate_plate(plate_text)
    
    # Additional validation for common OCR errors
    if not result['valid'] and plate_text:
        # Try common OCR corrections
        corrected = plate_text.upper().replace('0', 'O').replace('1', 'I')
        if corrected != plate_text.upper():
            result = validator.validate_plate(corrected)
            if result['valid']:
                result['ocr_corrected'] = True
                result['original_text'] = plate_text
    
    return result