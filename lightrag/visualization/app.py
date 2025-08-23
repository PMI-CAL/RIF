"""
LightRAG Knowledge Visualization Tool
Main Streamlit application for exploring and analyzing LightRAG data.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import json
import logging
from pathlib import Path

# Import our data access layer and components
from data_access import get_data_access
from components.dashboard import render_enhanced_overview_dashboard
from components.search import render_search_page_enhanced
from components.export import render_export_page
from components.statistics import render_statistics_page

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LightRAG Visualization",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_app():
    """Initialize the application and session state."""
    if 'data_access' not in st.session_state:
        try:
            st.session_state.data_access = get_data_access()
            st.session_state.app_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize data access: {e}")
            st.session_state.app_initialized = False
            return False
    
    if 'selected_collection' not in st.session_state:
        st.session_state.selected_collection = None
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    return st.session_state.app_initialized

def render_sidebar():
    """Render the application sidebar with navigation."""
    st.sidebar.title("🧠 LightRAG Viz")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a view",
        ["Overview", "Collections", "Search", "Document Browser", "Statistics", "Export"],
        key="navigation"
    )
    
    st.sidebar.markdown("---")
    
    # Quick stats if data is available
    if st.session_state.get('app_initialized'):
        try:
            overview = st.session_state.data_access.get_collection_overview()
            if 'error' not in overview:
                st.sidebar.metric("Total Collections", overview.get('total_collections', 0))
                st.sidebar.metric("Total Documents", overview.get('total_documents', 0))
        except Exception as e:
            st.sidebar.warning(f"Could not load stats: {e}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("Select different views from the dropdown above to explore your LightRAG knowledge base.")
    
    return page

def render_overview_page():
    """Render the main overview page with enhanced dashboard."""
    try:
        # Get collection overview
        overview = st.session_state.data_access.get_collection_overview()
        
        if 'error' in overview:
            st.error(f"Error loading overview: {overview['error']}")
            return
        
        # Use the enhanced dashboard component
        render_enhanced_overview_dashboard(overview)
        
        # Display raw overview data in expander for debugging
        with st.expander("🔍 Raw Data (Debug)", expanded=False):
            st.json(overview)
            
    except Exception as e:
        st.error(f"Failed to load overview: {e}")
        logger.error(f"Overview page error: {e}")

def render_collections_page():
    """Render the collections detail page."""
    st.title("Collection Details")
    
    try:
        # Get available collections
        overview = st.session_state.data_access.get_collection_overview()
        
        if 'error' in overview:
            st.error(f"Error loading collections: {overview['error']}")
            return
        
        collections = list(overview.get('collections', {}).keys())
        
        if not collections:
            st.warning("No collections found.")
            return
        
        # Collection selector
        selected_collection = st.selectbox(
            "Select a collection to explore:",
            collections,
            key="collection_selector"
        )
        
        if selected_collection:
            # Store in session state
            st.session_state.selected_collection = selected_collection
            
            # Get detailed collection info
            details = st.session_state.data_access.get_collection_details(selected_collection)
            
            if 'error' in details:
                st.error(f"Error loading collection details: {details['error']}")
                return
            
            # Display collection info
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Document Count", details.get('count', 0))
                st.metric("Metadata Fields", len(details.get('metadata_fields', [])))
            
            with col2:
                st.text_area(
                    "Description",
                    details.get('description', 'No description'),
                    height=100,
                    disabled=True
                )
            
            # Metadata fields
            if details.get('metadata_fields'):
                st.subheader("Available Metadata Fields")
                metadata_df = pd.DataFrame({
                    'Field': details['metadata_fields'],
                    'Type': ['string'] * len(details['metadata_fields'])  # Placeholder
                })
                st.dataframe(metadata_df, use_container_width=True)
            
            # Sample documents
            st.subheader("Sample Documents")
            sample_docs = details.get('sample_documents', [])
            
            if sample_docs:
                for i, doc in enumerate(sample_docs[:5]):  # Show first 5
                    with st.expander(f"Document: {doc.get('id', 'Unknown ID')}"):
                        st.text_area(
                            "Content Preview",
                            doc.get('content_preview', 'No content'),
                            height=100,
                            disabled=True,
                            key=f"sample_content_{i}"
                        )
                        
                        if doc.get('metadata'):
                            st.json(doc['metadata'])
            else:
                st.info("No sample documents available.")
    
    except Exception as e:
        st.error(f"Failed to load collection details: {e}")
        logger.error(f"Collections page error: {e}")

def render_search_page():
    """Render the enhanced search page."""
    render_search_page_enhanced(st.session_state.data_access)

def render_document_browser_page():
    """Render the document browser page."""
    st.title("Document Browser")
    
    try:
        # Collection selector
        overview = st.session_state.data_access.get_collection_overview()
        collections = list(overview.get('collections', {}).keys()) if 'error' not in overview else []
        
        if not collections:
            st.warning("No collections available for browsing.")
            return
        
        selected_collection = st.selectbox(
            "Select collection to browse:",
            collections,
            key="browser_collection"
        )
        
        if selected_collection:
            # Pagination controls
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                page_size = st.selectbox("Documents per page:", [10, 25, 50, 100], index=1)
            
            with col2:
                if 'current_page' not in st.session_state:
                    st.session_state.current_page = 0
                
                # Get current page data
                page_data = st.session_state.data_access.get_documents_paginated(
                    collection_name=selected_collection,
                    page=st.session_state.current_page,
                    page_size=page_size
                )
                
                if 'error' not in page_data:
                    pagination = page_data.get('pagination', {})
                    st.write(f"Page {pagination.get('page', 0) + 1} of {pagination.get('total_pages', 1)}")
            
            with col3:
                col3a, col3b = st.columns(2)
                with col3a:
                    if st.button("Previous") and st.session_state.current_page > 0:
                        st.session_state.current_page -= 1
                        st.rerun()
                
                with col3b:
                    if 'error' not in page_data:
                        pagination = page_data.get('pagination', {})
                        if st.button("Next") and pagination.get('has_next', False):
                            st.session_state.current_page += 1
                            st.rerun()
            
            # Display documents
            if 'error' in page_data:
                st.error(f"Error loading documents: {page_data['error']}")
            else:
                documents = page_data.get('documents', [])
                
                if documents:
                    for doc in documents:
                        with st.expander(f"Document: {doc.get('id', 'Unknown')}"):
                            st.text_area(
                                "Content",
                                doc.get('content', 'No content available'),
                                height=200,
                                disabled=True,
                                key=f"doc_content_{doc.get('id', 'unknown')}"
                            )
                            
                            if doc.get('metadata'):
                                st.subheader("Metadata")
                                st.json(doc['metadata'])
                else:
                    st.info("No documents found in this collection.")
    
    except Exception as e:
        st.error(f"Document browser failed: {e}")
        logger.error(f"Document browser error: {e}")

def render_statistics_page_wrapper():
    """Render the statistics page using the enhanced component."""
    render_statistics_page(st.session_state.data_access)

def render_export_page_wrapper():
    """Render the export page using the enhanced component."""
    render_export_page(st.session_state.data_access)

def main():
    """Main application entry point."""
    # Initialize the app
    if not init_app():
        st.error("Failed to initialize the application. Please check your LightRAG setup.")
        return
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render the appropriate page
    if page == "Overview":
        render_overview_page()
    elif page == "Collections":
        render_collections_page()
    elif page == "Search":
        render_search_page()
    elif page == "Document Browser":
        render_document_browser_page()
    elif page == "Statistics":
        render_statistics_page_wrapper()
    elif page == "Export":
        render_export_page_wrapper()
    else:
        st.error(f"Unknown page: {page}")

if __name__ == "__main__":
    main()