"""
Utility functions for LightRAG visualization tool.
"""

import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import streamlit as st


def format_timestamp(timestamp_str: str) -> str:
    """
    Format timestamp string for display.
    
    Args:
        timestamp_str: ISO timestamp string
        
    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return timestamp_str


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length].rstrip() + "..."


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string with fallback.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except Exception:
        return default


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def create_download_link(data: Any, filename: str, format_type: str = "json") -> str:
    """
    Create a download link for data.
    
    Args:
        data: Data to download
        filename: Name of file
        format_type: Format type (json, csv, txt)
        
    Returns:
        Download link HTML
    """
    try:
        if format_type == "json":
            content = json.dumps(data, indent=2)
            mime_type = "application/json"
        elif format_type == "csv":
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                content = df.to_csv(index=False)
            else:
                content = str(data)
            mime_type = "text/csv"
        else:
            content = str(data)
            mime_type = "text/plain"
        
        return st.download_button(
            label=f"Download {filename}",
            data=content,
            file_name=filename,
            mime=mime_type
        )
    except Exception as e:
        st.error(f"Failed to create download link: {e}")
        return ""


def validate_collection_name(name: str) -> Tuple[bool, str]:
    """
    Validate collection name.
    
    Args:
        name: Collection name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, "Collection name cannot be empty"
    
    if len(name) > 100:
        return False, "Collection name too long (max 100 characters)"
    
    # Add more validation rules as needed
    return True, ""


def calculate_similarity_score(distance: float) -> float:
    """
    Convert distance to similarity score.
    
    Args:
        distance: Distance value from vector search
        
    Returns:
        Similarity score (0-1, higher is more similar)
    """
    if distance is None:
        return 0.0
    
    # Convert distance to similarity (assuming cosine distance)
    return max(0.0, 1.0 - distance)


def format_metadata_for_display(metadata: Dict[str, Any]) -> str:
    """
    Format metadata dictionary for display.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Formatted string
    """
    if not metadata:
        return "No metadata"
    
    formatted_items = []
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, indent=2)
        else:
            value_str = str(value)
        
        formatted_items.append(f"**{key}**: {value_str}")
    
    return "\n".join(formatted_items)


def extract_keywords_from_content(content: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from content (simple implementation).
    
    Args:
        content: Text content
        max_keywords: Maximum number of keywords
        
    Returns:
        List of keywords
    """
    if not content:
        return []
    
    # Simple keyword extraction (word frequency)
    import re
    from collections import Counter
    
    # Clean and tokenize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'under', 'between', 'among', 'this', 'that',
        'these', 'those', 'is', 'are', 'was', 'were', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
    }
    
    filtered_words = [word for word in words if word not in stop_words]
    
    # Get most common words
    word_counts = Counter(filtered_words)
    keywords = [word for word, count in word_counts.most_common(max_keywords)]
    
    return keywords


def create_collection_summary(collection_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a summary of collection data.
    
    Args:
        collection_data: Collection data dictionary
        
    Returns:
        Summary dictionary
    """
    summary = {
        "total_documents": 0,
        "metadata_fields": set(),
        "content_types": set(),
        "date_range": {"earliest": None, "latest": None},
        "size_stats": {"min_content": float('inf'), "max_content": 0, "avg_content": 0}
    }
    
    documents = collection_data.get("documents", [])
    if not documents:
        return summary
    
    summary["total_documents"] = len(documents)
    
    content_lengths = []
    dates = []
    
    for doc in documents:
        # Analyze metadata
        metadata = doc.get("metadata", {})
        if metadata:
            summary["metadata_fields"].update(metadata.keys())
            
            # Extract content type if available
            if "type" in metadata:
                summary["content_types"].add(metadata["type"])
            
            # Extract timestamps
            for key in ["timestamp", "created_at", "updated_at"]:
                if key in metadata:
                    try:
                        date = datetime.fromisoformat(metadata[key].replace('Z', '+00:00'))
                        dates.append(date)
                    except Exception:
                        pass
        
        # Analyze content
        content = doc.get("content", "")
        if content:
            content_lengths.append(len(content))
    
    # Calculate content statistics
    if content_lengths:
        summary["size_stats"]["min_content"] = min(content_lengths)
        summary["size_stats"]["max_content"] = max(content_lengths)
        summary["size_stats"]["avg_content"] = sum(content_lengths) / len(content_lengths)
    
    # Calculate date range
    if dates:
        summary["date_range"]["earliest"] = min(dates).isoformat()
        summary["date_range"]["latest"] = max(dates).isoformat()
    
    # Convert sets to lists for JSON serialization
    summary["metadata_fields"] = list(summary["metadata_fields"])
    summary["content_types"] = list(summary["content_types"])
    
    return summary


class PerformanceMonitor:
    """Simple performance monitoring for operations."""
    
    def __init__(self):
        self.operations = {}
    
    def start_operation(self, name: str):
        """Start timing an operation."""
        self.operations[name] = {"start": datetime.now()}
    
    def end_operation(self, name: str) -> float:
        """End timing an operation and return duration."""
        if name in self.operations:
            duration = (datetime.now() - self.operations[name]["start"]).total_seconds()
            self.operations[name]["duration"] = duration
            return duration
        return 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return {name: op.get("duration", 0.0) for name, op in self.operations.items()}


# Global performance monitor instance
perf_monitor = PerformanceMonitor()


def log_operation_time(operation_name: str):
    """Decorator to log operation time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            perf_monitor.start_operation(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = perf_monitor.end_operation(operation_name)
                logging.info(f"Operation '{operation_name}' took {duration:.2f} seconds")
        return wrapper
    return decorator