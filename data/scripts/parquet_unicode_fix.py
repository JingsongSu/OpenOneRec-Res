#!/usr/bin/env python3
"""Parquet Unicode Fix Script

Fix unicode Chinese garbled text issues in messages and segments fields of parquet files.
Supports single file or batch directory processing.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def decode_unicode_json(json_str: Optional[Union[str, bytes]]) -> Optional[str]:
    """Decode unicode characters in JSON string.

    Args:
        json_str: JSON string that may contain unicode encoding

    Returns:
        Decoded JSON string
    """
    if json_str is None or pd.isna(json_str):
        return json_str

    # Handle bytes type
    if isinstance(json_str, bytes):
        json_str = json_str.decode('utf-8', errors='ignore')

    # If already a string and doesn't contain unicode escape sequences, return directly
    if isinstance(json_str, str) and '\\u' not in json_str:
        return json_str
    
    try:
        # JSON load (automatically decode unicode)
        json_obj = json.loads(json_str)

        # JSON dump with ensure_ascii disabled (preserve Chinese characters)
        decoded_str = json.dumps(
            json_obj,
            ensure_ascii=False,  # Key: don't convert Chinese to unicode
            indent=None,         # Keep original compact format
            separators=(',', ':')  # Keep original separator format
        )
        return decoded_str

    except json.JSONDecodeError:
        # Return original string when JSON parsing fails
        return json_str
    except Exception as e:
        logger.debug(f"Error processing JSON string: {e}")
        return json_str

def find_parquet_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Find all parquet files in the directory

    Args:
        directory: Directory path
        recursive: Whether to recursively search subdirectories, default True

    Returns:
        List of parquet file paths
    """
    parquet_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    pattern = "**/*.parquet" if recursive else "*.parquet"
    parquet_files = [str(p) for p in directory_path.glob(pattern) if p.is_file()]

    logger.info(f"Found {len(parquet_files)} parquet files in directory {directory}")
    return sorted(parquet_files)

def get_output_path(input_path: str, output_base: str, input_base: Optional[str] = None) -> str:
    """
    Generate output path based on input path and output base path

    Args:
        input_path: Input file path
        output_base: Output base path (file or directory)
        input_base: Input base path (to maintain relative path structure), if None uses input file's directory

    Returns:
        Output file path
    """
    input_path_obj = Path(input_path)
    output_base_obj = Path(output_base)

    # If output base path is a file, return directly
    if output_base_obj.is_file() or (not output_base_obj.exists() and not output_base_obj.suffix == ''):
        return str(output_base_obj)

    # If output base path is a directory
    if input_base:
        # Maintain relative path structure
        input_base_obj = Path(input_base)
        try:
            relative_path = input_path_obj.relative_to(input_base_obj)
            output_path = output_base_obj / relative_path
        except ValueError:
            # If unable to calculate relative path, use filename
            output_path = output_base_obj / input_path_obj.name
    else:
        # Use input file's directory as base
        output_path = output_base_obj / input_path_obj.name

    return str(output_path)

def process_parquet_file(
    input_path: str,
    output_path: str,
    engine: str = 'pyarrow',
    fields: Optional[List[str]] = None
) -> None:
    """Process parquet file to fix unicode Chinese garbled text in specified fields.

    Args:
        input_path: Input parquet file path
        output_path: Output parquet file path
        engine: Engine for reading/writing parquet, options: 'pyarrow' or 'fastparquet'
        fields: List of fields to process, defaults to ['messages', 'segments']
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if fields is None:
        fields = ['messages', 'segments']

    # Read parquet file
    logger.info(f"Reading file: {input_path}")
    df = pd.read_parquet(input_path, engine=engine)
    logger.info(f"Total rows: {len(df)}")
    
    # Check and process fields
    processed_fields = []
    for field in fields:
        if field in df.columns:
            logger.debug(f"Processing field: {field}")
            df[field] = df[field].apply(decode_unicode_json)
            processed_fields.append(field)
        else:
            logger.debug(f"Field does not exist, skipping: {field}")
    
    if not processed_fields:
        logger.warning(f"No fields to process found: {fields}")
        # If no fields to process, copy file directly
        if input_path != output_path:
            import shutil
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            logger.info(f"File copied to: {output_path}")
        return

    logger.info(f"Processed fields: {processed_fields}")

    # Save processed file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        output_path,
        engine=engine,
        index=False,
        compression='snappy'
    )
    logger.info(f"File saved successfully: {output_path}")

def process_directory(input_dir: str, output_dir: str, engine: str = 'pyarrow', recursive: bool = True, overwrite: bool = False) -> None:
    """
    Batch process all parquet files in the directory

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        engine: Parquet processing engine
        recursive: Whether to recursively process subdirectories
        overwrite: Whether to overwrite original files (if True, output_dir is ignored and input files are overwritten directly)
    """
    # Find all parquet files
    parquet_files = find_parquet_files(input_dir, recursive=recursive)

    if not parquet_files:
        logger.warning(f"No parquet files found in directory {input_dir}")
        return

    # Create output directory (if needed)
    if not overwrite:
        output_path_obj = Path(output_dir)
        output_path_obj.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
    # Process each file
    total_files = len(parquet_files)
    success_count = 0
    fail_count = 0

    for input_file in tqdm(parquet_files, desc="Processing files"):
        try:
            if overwrite:
                # Overwrite original file
                output_file = input_file
            else:
                # Generate output path, maintain directory structure
                output_file = get_output_path(input_file, output_dir, input_dir)
                # Ensure output directory exists
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            process_parquet_file(input_file, output_file, engine)
            success_count += 1

        except Exception as e:
            fail_count += 1
            logger.error(f"File processing failed: {input_file}, error: {e}", exc_info=True)
            continue

    # Output statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch processing completed!")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"{'='*60}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process unicode Chinese garbled text in messages and segments fields of parquet files (supports single file or batch directory processing)'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input parquet file path or directory path (required)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output parquet file path or directory path (required)'
    )
    parser.add_argument(
        '-e', '--engine',
        choices=['pyarrow', 'fastparquet'],
        default='pyarrow',
        help='Parquet processing engine, default uses pyarrow'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='When processing directory, do not recursively process subdirectories (only process files in current directory)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite original files (only effective when input is directory, will ignore output path)'
    )
    
    args = parser.parse_args()

    # Execute processing
    try:
        input_path = Path(args.input)

        if not input_path.exists():
            logger.error(f"Input path does not exist: {args.input}")
            exit(1)
        
        # Determine if input is file or directory
        if input_path.is_file():
            # Single file processing mode
            logger.info("Single file processing mode")
            if Path(args.output).is_dir():
                # If output is directory, create file with same name in directory
                output_file = Path(args.output) / input_path.name
            else:
                output_file = args.output
            
            process_parquet_file(
                input_path=str(input_path),
                output_path=str(output_file),
                engine=args.engine
            )
            logger.info("All operations completed!")

        elif input_path.is_dir():
            # Directory batch processing mode
            logger.info("Directory batch processing mode")
            if args.overwrite:
                logger.info("Will overwrite original files")
                process_directory(
                    input_dir=str(input_path),
                    output_dir="",  # Will not be used
                    engine=args.engine,
                    recursive=not args.no_recursive,
                    overwrite=True
                )
            else:
                output_path = Path(args.output)
                if output_path.exists() and output_path.is_file():
                    logger.error(f"When input is directory, output should also be directory, but output path is file: {args.output}")
                    exit(1)
                process_directory(
                    input_dir=str(input_path),
                    output_dir=str(output_path),
                    engine=args.engine,
                    recursive=not args.no_recursive,
                    overwrite=False
                )
            logger.info("All operations completed!")
        else:
            logger.error(f"Input path is neither file nor directory: {args.input}")
            exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
