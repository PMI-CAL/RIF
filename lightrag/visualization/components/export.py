"""
Export functionality for LightRAG visualization.
Handles data export in multiple formats with filtering options.
"""

import streamlit as st
import json
import csv
import io
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def render_export_interface(collections: List[str], data_access) -> None:
    """
    Render the export interface with options and controls.
    
    Args:
        collections: List of available collection names
        data_access: Data access instance for export operations
    """
    st.subheader("📤 Export Data")
    st.markdown("*Export your LightRAG data in various formats*")
    
    # Export configuration
    col1, col2 = st.columns(2)
    
    with col1:
        # Collection selection
        export_collection = st.selectbox(
            "Select Collection to Export:",
            collections,
            help="Choose which collection to export"
        )
        
        # Export format
        export_format = st.selectbox(
            "Export Format:",
            ["JSON", "CSV", "Text"],
            help="Choose the export format"
        )
    
    with col2:
        # Export options
        st.write("**Export Options:**")
        
        include_metadata = st.checkbox(
            "Include Metadata",
            value=True,
            help="Include document metadata in export"
        )
        
        include_ids = st.checkbox(
            "Include Document IDs",
            value=True,
            help="Include document IDs in export"
        )
        
        pretty_format = st.checkbox(
            "Pretty Format",
            value=True,
            help="Format output for readability"
        )
    
    # Advanced filters
    with st.expander("🔧 Advanced Export Filters", expanded=False):
        st.markdown("*Configure what data to include in your export*")
        
        # Content length filter
        col1, col2 = st.columns(2)
        
        with col1:
            min_content_length = st.number_input(
                "Minimum Content Length",
                min_value=0,
                value=0,
                help="Exclude documents shorter than this"
            )
        
        with col2:
            max_content_length = st.number_input(
                "Maximum Content Length",
                min_value=0,
                value=0,
                help="Exclude documents longer than this (0 = no limit)"
            )
        
        # Metadata filters
        st.write("**Metadata Filters:**")
        metadata_filter_key = st.text_input(
            "Metadata Key",
            placeholder="e.g., type, status, category",
            help="Filter by metadata key"
        )
        
        metadata_filter_value = st.text_input(
            "Metadata Value", 
            placeholder="e.g., pattern, active, important",
            help="Required value for the metadata key"
        )
        
        # Date range filter (placeholder)
        st.info("📅 Date range filtering will be available in Phase 3")
    
    # Export preview
    if st.button("🔍 Preview Export", type="secondary"):
        with st.spinner("Generating preview..."):
            preview_data = generate_export_preview(
                data_access,
                export_collection,
                export_format,
                {
                    'include_metadata': include_metadata,
                    'include_ids': include_ids,
                    'pretty_format': pretty_format,
                    'min_content_length': min_content_length,
                    'max_content_length': max_content_length,
                    'metadata_filter_key': metadata_filter_key,
                    'metadata_filter_value': metadata_filter_value
                }
            )
            
            if preview_data:
                render_export_preview(preview_data, export_format)
    
    # Export button
    if st.button("📥 Export Data", type="primary"):
        with st.spinner("Preparing export..."):
            export_data = generate_full_export(
                data_access,
                export_collection,
                export_format,
                {
                    'include_metadata': include_metadata,
                    'include_ids': include_ids,
                    'pretty_format': pretty_format,
                    'min_content_length': min_content_length,
                    'max_content_length': max_content_length,
                    'metadata_filter_key': metadata_filter_key,
                    'metadata_filter_value': metadata_filter_value
                }
            )
            
            if export_data:
                render_download_button(export_data, export_collection, export_format)


def generate_export_preview(data_access, collection_name: str, format_type: str, options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate a preview of the export data.
    
    Args:
        data_access: Data access instance
        collection_name: Name of collection to export
        format_type: Export format type
        options: Export options dictionary
        
    Returns:
        Preview data or None if error
    """
    try:
        # Get sample data (first 5 documents)
        page_data = data_access.get_documents_paginated(
            collection_name=collection_name,
            page=0,
            page_size=5
        )
        
        if 'error' in page_data:
            st.error(f"Error generating preview: {page_data['error']}")
            return None
        
        documents = page_data.get('documents', [])
        
        # Apply filters
        filtered_docs = apply_export_filters(documents, options)
        
        # Format for preview
        formatted_data = format_export_data(filtered_docs, format_type, options)
        
        return {
            'data': formatted_data,
            'count': len(filtered_docs),
            'total_available': page_data.get('pagination', {}).get('total_count', 0),
            'format': format_type
        }
        
    except Exception as e:
        logger.error(f"Export preview error: {e}")
        st.error(f"Failed to generate preview: {e}")
        return None


def generate_full_export(data_access, collection_name: str, format_type: str, options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate the full export data.
    
    Args:
        data_access: Data access instance
        collection_name: Name of collection to export
        format_type: Export format type
        options: Export options dictionary
        
    Returns:
        Export data or None if error
    """
    try:
        # Use the data access export functionality
        export_result = data_access.export_collection(
            collection_name=collection_name,
            format=format_type.lower()
        )
        
        if 'error' in export_result:
            st.error(f"Export error: {export_result['error']}")
            return None
        
        # Get the raw data
        if 'data' not in export_result:
            st.error("No data in export result")
            return None
        
        raw_data = export_result['data']
        
        # Extract documents for filtering
        if isinstance(raw_data, dict) and 'documents' in raw_data:
            documents = raw_data['documents']
        elif isinstance(raw_data, list):
            documents = raw_data
        else:
            st.error("Unexpected export data format")
            return None
        
        # Apply filters
        filtered_docs = apply_export_filters(documents, options)
        
        # Format the data
        formatted_data = format_export_data(filtered_docs, format_type, options)
        
        return {
            'data': formatted_data,
            'count': len(filtered_docs),
            'format': format_type,
            'collection': collection_name,
            'exported_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Full export error: {e}")
        st.error(f"Failed to generate export: {e}")
        return None


def apply_export_filters(documents: List[Dict[str, Any]], options: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply filters to documents before export.
    
    Args:
        documents: List of document dictionaries
        options: Filter options
        
    Returns:
        Filtered list of documents
    """
    filtered_docs = []
    
    for doc in documents:
        # Content length filters
        content = doc.get('content', '')
        content_length = len(content)
        
        if options.get('min_content_length', 0) > 0:
            if content_length < options['min_content_length']:
                continue
        
        if options.get('max_content_length', 0) > 0:
            if content_length > options['max_content_length']:
                continue
        
        # Metadata filters
        metadata_key = options.get('metadata_filter_key', '').strip()
        metadata_value = options.get('metadata_filter_value', '').strip()
        
        if metadata_key and metadata_value:
            metadata = doc.get('metadata', {})
            if metadata_key not in metadata or str(metadata[metadata_key]) != metadata_value:
                continue
        
        filtered_docs.append(doc)
    
    return filtered_docs


def format_export_data(documents: List[Dict[str, Any]], format_type: str, options: Dict[str, Any]) -> str:
    """
    Format documents for export based on format type.
    
    Args:
        documents: List of document dictionaries
        format_type: Export format type
        options: Formatting options
        
    Returns:
        Formatted export string
    """
    if format_type.upper() == "JSON":
        return format_json_export(documents, options)
    elif format_type.upper() == "CSV":
        return format_csv_export(documents, options)
    elif format_type.upper() == "TEXT":
        return format_text_export(documents, options)
    else:
        return str(documents)


def format_json_export(documents: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
    """Format documents as JSON."""
    export_data = {
        'exported_at': datetime.utcnow().isoformat(),
        'count': len(documents),
        'documents': []
    }
    
    for doc in documents:
        export_doc = {}
        
        if options.get('include_ids', True):
            export_doc['id'] = doc.get('id', '')
        
        export_doc['content'] = doc.get('content', '')
        
        if options.get('include_metadata', True):
            export_doc['metadata'] = doc.get('metadata', {})
        
        export_data['documents'].append(export_doc)
    
    if options.get('pretty_format', True):
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    else:
        return json.dumps(export_data, ensure_ascii=False)


def format_csv_export(documents: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
    """Format documents as CSV."""
    output = io.StringIO()
    
    # Determine columns
    columns = []
    if options.get('include_ids', True):
        columns.append('id')
    columns.append('content')
    if options.get('include_metadata', True):
        columns.append('metadata')
    
    writer = csv.writer(output)
    writer.writerow(columns)
    
    for doc in documents:
        row = []
        
        if options.get('include_ids', True):
            row.append(doc.get('id', ''))
        
        # Clean content for CSV
        content = doc.get('content', '').replace('\n', ' ').replace('\r', ' ')
        row.append(content)
        
        if options.get('include_metadata', True):
            metadata_str = json.dumps(doc.get('metadata', {}))
            row.append(metadata_str)
        
        writer.writerow(row)
    
    return output.getvalue()


def format_text_export(documents: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
    """Format documents as plain text."""
    lines = []
    lines.append(f"# LightRAG Export - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# Total Documents: {len(documents)}")
    lines.append("")
    
    for i, doc in enumerate(documents, 1):
        lines.append(f"## Document {i}")
        
        if options.get('include_ids', True):
            lines.append(f"**ID:** {doc.get('id', 'Unknown')}")
        
        lines.append("**Content:**")
        lines.append(doc.get('content', ''))
        
        if options.get('include_metadata', True):
            metadata = doc.get('metadata', {})
            if metadata:
                lines.append("**Metadata:**")
                for key, value in metadata.items():
                    lines.append(f"  - {key}: {value}")
        
        lines.append("")
        lines.append("-" * 50)
        lines.append("")
    
    return "\n".join(lines)


def render_export_preview(preview_data: Dict[str, Any], format_type: str) -> None:
    """
    Render the export preview.
    
    Args:
        preview_data: Preview data dictionary
        format_type: Export format type
    """
    st.subheader("📋 Export Preview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Preview Documents", preview_data['count'])
    
    with col2:
        st.metric("Total Available", preview_data['total_available'])
    
    with col3:
        st.metric("Format", format_type)
    
    # Show preview content
    preview_content = preview_data['data']
    
    if format_type.upper() == "JSON":
        st.code(preview_content[:2000] + "..." if len(preview_content) > 2000 else preview_content, language="json")
    elif format_type.upper() == "CSV":
        st.code(preview_content[:2000] + "..." if len(preview_content) > 2000 else preview_content, language="csv")
    else:
        st.text_area("Preview", preview_content[:2000] + "..." if len(preview_content) > 2000 else preview_content, height=300)
    
    if len(preview_content) > 2000:
        st.info("Preview truncated. Full export will contain all data.")


def render_download_button(export_data: Dict[str, Any], collection_name: str, format_type: str) -> None:
    """
    Render the download button for export data.
    
    Args:
        export_data: Export data dictionary
        collection_name: Name of exported collection
        format_type: Export format type
    """
    # Generate filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    extension = format_type.lower()
    filename = f"lightrag_{collection_name}_{timestamp}.{extension}"
    
    # Determine MIME type
    mime_types = {
        'JSON': 'application/json',
        'CSV': 'text/csv',
        'TEXT': 'text/plain'
    }
    mime_type = mime_types.get(format_type.upper(), 'text/plain')
    
    # Show export summary
    st.success(f"✅ Export ready! {export_data['count']} documents prepared.")
    
    # Download button
    st.download_button(
        label=f"📥 Download {format_type} Export",
        data=export_data['data'],
        file_name=filename,
        mime=mime_type,
        type="primary"
    )
    
    # Export info
    with st.expander("📊 Export Details"):
        st.write(f"**Collection:** {collection_name}")
        st.write(f"**Format:** {format_type}")
        st.write(f"**Document Count:** {export_data['count']:,}")
        st.write(f"**Generated:** {export_data['exported_at']}")
        st.write(f"**Filename:** {filename}")


def render_export_page(data_access) -> None:
    """
    Render the complete export page.
    
    Args:
        data_access: Data access instance
    """
    st.title("📤 Data Export")
    st.markdown("*Export your LightRAG knowledge base data in multiple formats*")
    
    try:
        # Get available collections
        overview = data_access.get_collection_overview()
        
        if 'error' in overview:
            st.error(f"Error loading collections: {overview['error']}")
            return
        
        collections = list(overview.get('collections', {}).keys())
        
        if not collections:
            st.warning("No collections available for export.")
            return
        
        # Render export interface
        render_export_interface(collections, data_access)
        
        # Export tips
        with st.expander("💡 Export Tips", expanded=False):
            st.markdown("""
            ### Export Tips & Best Practices
            
            **Format Recommendations:**
            - **JSON**: Best for preserving metadata and structure, ideal for backup and data migration
            - **CSV**: Great for spreadsheet analysis and data processing tools
            - **Text**: Human-readable format, perfect for documentation and review
            
            **Performance Tips:**
            - Large collections may take time to export
            - Use filters to reduce export size
            - Preview before full export to verify settings
            
            **File Size Considerations:**
            - JSON exports are typically largest due to structure
            - CSV exports are medium-sized and efficient
            - Text exports vary based on content structure
            
            **Data Usage:**
            - Exported data includes all selected metadata
            - Document IDs are preserved for reimport
            - Timestamps show export generation time
            """)
    
    except Exception as e:
        st.error(f"Export page error: {e}")
        logger.error(f"Export page error: {e}")