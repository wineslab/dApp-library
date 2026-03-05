#!/usr/bin/env python3
"""
ASN.1 to JSON Schema Transpiler
Uses asn1tools parse_files to get full AST with constraints.
Inspired by the Erlang cgf_tools module for the CGF (https://github.com/sigscale/cgf/blob/main/src/cgf_tools.erl)
"""

__author__ = "Andrea Lacava"

import asn1tools
import json
from pathlib import Path

class ASN1ToJSONSchemaTranspiler:
    
    def __init__(self, generic=False):
        self.all_types = set()  # Store all defined type names
        self.constants = {}     # Store constant values (e.g., maxPRBs ::= 273)
        self.generic = generic  # If True, OCTET STRING is generic object; if False, oneOf with hex or object
        
    def transpile(self, asn1_file, json_schema_file=None):
        """Convert ASN.1 file to JSON Schema"""
        
        if json_schema_file is None:
            json_schema_file = Path(asn1_file).with_suffix('.json')
        
        # Parse ASN.1 file to get raw AST with constraints
        try:
            parsed = asn1tools.parse_files([asn1_file])
        except Exception as e:
            return {'error': f'Failed to parse ASN.1: {e}'}
        
        # Get the first (usually only) module
        if not parsed:
            return {'error': 'No modules found in ASN.1 file'}
        
        module = list(parsed.values())[0]
        types_dict = module.get('types', {})
        values_dict = module.get('values', {})
        
        # Store all type names for reference resolution
        self.all_types = set(types_dict.keys())
        
        # Store constant values for constraint resolution
        for const_name, const_def in values_dict.items():
            if isinstance(const_def, dict) and 'value' in const_def:
                self.constants[const_name] = const_def['value']
            elif isinstance(const_def, (int, float)):
                self.constants[const_name] = const_def
        
        # Build JSON Schema
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "_copyright": "Copyright (c) 2026, Andrea Lacava.  All rights reserved.",
            "$defs": {}
        }
        
        # Convert each type
        for type_name, type_def in types_dict.items():
            schema["$defs"][type_name] = self.convert_type(type_def)
        
        # Write output
        with open(json_schema_file, 'w') as f:
            json.dump(schema, f, indent=2)
        
        return 'ok'
    
    def convert_type(self, type_def):
        """Convert ASN.1 type definition (dict) to JSON Schema"""
        
        if not isinstance(type_def, dict):
            return {'description': f'Invalid type definition: {type_def}'}
        
        type_name = type_def.get('type', '')
        
        # Check if this is a reference to a defined type
        if type_name in self.all_types:
            return {'$ref': f"#/$defs/{type_name}"}
        
        # Basic types
        if type_name == 'BOOLEAN':
            return {'type': 'boolean'}
        elif type_name == 'INTEGER':
            return self.convert_integer(type_def)
        elif type_name == 'REAL':
            return {'type': 'number'}
        elif type_name == 'NULL':
            return {'type': 'null'}
        elif type_name in ('OCTET STRING', 'BIT STRING', 'IA5String', 
                           'VisibleString', 'NumericString', 'UTF8String',
                           'PrintableString', 'TeletexString', 'BMPString',
                           'GeneralString', 'GraphicString'):
            return self.convert_string_type(type_def, type_name)
        elif type_name == 'SEQUENCE':
            return self.convert_sequence(type_def)
        elif type_name == 'SEQUENCE OF':
            return self.convert_sequence_of(type_def)
        elif type_name == 'SET':
            return self.convert_sequence(type_def)  # Sets are similar to sequences
        elif type_name == 'SET OF':
            return self.convert_sequence_of(type_def)
        elif type_name == 'CHOICE':
            return self.convert_choice(type_def)
        elif type_name == 'ENUMERATED':
            return self.convert_enumerated(type_def)
        elif type_name == 'ANY':
            return {}  # Any type
        else:
            # Unknown type - return empty schema
            return {'description': f'Unknown ASN.1 type: {type_name}'}
    
    def resolve_value(self, value):
        """Resolve a value that may be a constant reference"""
        if isinstance(value, str) and value in self.constants:
            return self.constants[value]
        return value
    
    def convert_integer(self, type_def):
        """Convert INTEGER type with constraints"""
        schema = {'type': 'integer'}
        
        # Extract constraints from 'restricted-to' field
        restricted_to = type_def.get('restricted-to', [])
        if restricted_to and len(restricted_to) > 0:
            # restricted-to is a list of tuples (min, max)
            constraint = restricted_to[0]
            if isinstance(constraint, tuple) and len(constraint) == 2:
                schema['minimum'] = self.resolve_value(constraint[0])
                schema['maximum'] = self.resolve_value(constraint[1])
        
        return schema
    
    def convert_string_type(self, type_def, type_name):
        """Convert string types"""
        # OCTET STRING gets special handling - can be either hex string or decoded object
        if type_name == 'OCTET STRING':
            return self.convert_octet_string(type_def)
        
        schema = {'type': 'string'}
        
        if type_name == 'BIT STRING':
            schema['pattern'] = '^[01]*$'
        elif type_name == 'IA5String':
            schema['pattern'] = '^[\\x00-\\x7F]*$'
        elif type_name == 'VisibleString':
            schema['pattern'] = '^[\\x20-\\x7E]*$'
        elif type_name == 'NumericString':
            schema['pattern'] = '^[0-9 ]*$'
        elif type_name == 'PrintableString':
            schema['pattern'] = "^[A-Za-z0-9 '()+,-./:=?]*$"
        
        # Extract size constraints
        size = type_def.get('size', [])
        if size and len(size) > 0:
            constraint = size[0]
            if isinstance(constraint, tuple) and len(constraint) == 2:
                schema['minLength'] = self.resolve_value(constraint[0])
                schema['maxLength'] = self.resolve_value(constraint[1])
            elif isinstance(constraint, (int, str)):
                resolved = self.resolve_value(constraint)
                schema['minLength'] = resolved
                schema['maxLength'] = resolved
            
        return schema
    
    def convert_octet_string(self, type_def):
        """Convert OCTET STRING based on generic flag"""
        if self.generic:
            # Generic mode: OCTET STRING is just a generic object
            return {
                'type': 'object',
                'description': 'Generic object representation'
            }
        
        # Default mode: OCTET STRING is hex-encoded byte string
        hex_schema = {
            'type': 'string',
            'contentEncoding': 'base16',
            'pattern': '^([0-9a-fA-F]{2})*$'
        }
        
        # Extract size constraints for the hex string
        size = type_def.get('size', [])
        if size and len(size) > 0:
            constraint = size[0]
            if isinstance(constraint, tuple) and len(constraint) == 2:
                min_bytes = self.resolve_value(constraint[0])
                max_bytes = self.resolve_value(constraint[1])
                # Each byte is represented by 2 hex characters in the string
                hex_schema['minLength'] = min_bytes * 2  # 2 hex chars per byte
                hex_schema['maxLength'] = max_bytes * 2  # 2 hex chars per byte
            elif isinstance(constraint, (int, str)):
                resolved = self.resolve_value(constraint)
                # Each byte is represented by 2 hex characters in the string
                hex_schema['minLength'] = resolved * 2  # 2 hex chars per byte
                hex_schema['maxLength'] = resolved * 2  # 2 hex chars per byte
        
        return hex_schema
    
    def convert_sequence(self, type_def):
        """Convert SEQUENCE to object"""
        properties = {}
        required = []
        
        members = type_def.get('members', [])
        
        for member in members:
            if not isinstance(member, dict):
                continue
            member_name = member.get('name', '')
            if not member_name:
                continue
            
            prop_schema = self.convert_type(member)
            properties[member_name] = prop_schema
            
            if not member.get('optional', False):
                required.append(member_name)
        
        schema = {'type': 'object', 'properties': properties}
        if required:
            schema['required'] = required
            
        return schema
    
    def convert_sequence_of(self, type_def):
        """Convert SEQUENCE OF to array"""
        schema = {'type': 'array'}
        
        element = type_def.get('element')
        if element is not None:
            # Check if the element has 'restricted-to' constraint
            # This happens when the constraint is on the SEQUENCE OF itself
            # e.g., SEQUENCE OF SomeType (0..255)
            # asn1tools places it as element: {'type': 'SomeType', 'restricted-to': [(0, 255)]}
            if isinstance(element, dict) and 'restricted-to' in element:
                restricted_to = element.get('restricted-to', [])
                if restricted_to and len(restricted_to) > 0:
                    constraint = restricted_to[0]
                    if isinstance(constraint, tuple) and len(constraint) == 2:
                        schema['minItems'] = self.resolve_value(constraint[0])
                        schema['maxItems'] = self.resolve_value(constraint[1])
                
                # Convert the element type without the constraint
                element_without_constraint = element.copy()
                element_without_constraint.pop('restricted-to', None)
                schema['items'] = self.convert_type(element_without_constraint)
            else:
                schema['items'] = self.convert_type(element)
        
        # Also check for size constraints on the SEQUENCE OF itself
        size = type_def.get('size', [])
        if size and len(size) > 0:
            constraint = size[0]
            if isinstance(constraint, tuple) and len(constraint) == 2:
                schema['minItems'] = self.resolve_value(constraint[0])
                schema['maxItems'] = self.resolve_value(constraint[1])
            elif isinstance(constraint, (int, str)):
                resolved = self.resolve_value(constraint)
                schema['minItems'] = resolved
                schema['maxItems'] = resolved
                
        return schema
    
    def convert_choice(self, type_def):
        """Convert CHOICE to oneOf"""
        choices = []
        members = type_def.get('members', [])
        
        for member in members:
            if not isinstance(member, dict):
                continue
            member_name = member.get('name', '')
            if not member_name:
                continue
                
            choice = {
                'type': 'object',
                'properties': {
                    member_name: self.convert_type(member)
                },
                'required': [member_name],
                'additionalProperties': False
            }
            choices.append(choice)
        
        return {'oneOf': choices}
    
    def convert_enumerated(self, type_def):
        """Convert ENUMERATED to enum"""
        values = type_def.get('values', {})
        # values is a dict like {'insert': 0, 'update': 1, 'delete': 2}
        if isinstance(values, dict):
            enum_values = list(values.keys())
        elif isinstance(values, list):
            # Sometimes it's a list of tuples
            enum_values = [v[0] if isinstance(v, tuple) else v for v in values]
        else:
            enum_values = []
            
        return {
            'type': 'string',
            'enum': enum_values
        }


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert ASN.1 file to JSON Schema',
        epilog='Example: python asntojson.py /path/to/schema.asn'
    )
    parser.add_argument('asn_file', help='Path to the ASN.1 file to convert')
    parser.add_argument('-o', '--output', help='Output JSON schema file (default: same path with .json extension)')
    parser.add_argument('--generic', action='store_true', 
                        help='Generic mode: OCTET STRING fields are generic objects (no hex encoding option)')
    
    args = parser.parse_args()
    
    # Determine output path
    asn_path = Path(args.asn_file)
    if args.output:
        json_path = args.output
    else:
        json_path = asn_path.with_suffix('.json')
    
    transpiler = ASN1ToJSONSchemaTranspiler(generic=args.generic)
    result = transpiler.transpile(str(asn_path), str(json_path))
    
    if result == 'ok':
        print(f"Successfully converted {asn_path} -> {json_path}")
    else:
        print(f"Transpilation failed: {result}")