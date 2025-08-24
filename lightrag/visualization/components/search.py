"""
Enhanced search components for LightRAG visualization.
Advanced search interface with filtering and result display.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
import json
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def render_search_interface(collections: List[str]) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Render the enhanced search interface.
    
    Args:
        collections: List of available collection names
        
    Returns:
        Tuple of (search_query, selected_collection, filters)
    """
    st.subheader("🔍 Advanced Search")
    
    # Main search input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search Query",
            placeholder="Enter your search terms...",
            help="Search across document content using natural language",
            key="search_query"
        )
    
    with col2:
        search_button = st.button(
            "🔍 Search",
            type="primary",
            use_container_width=True
        )
    
    # Search options
    with st.expander("⚙️ Search Options", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Collection filter
            collection_options = ["All Collections"] + collections
            selected_collection_idx = st.selectbox(
                "Search in:",
                range(len(collection_options)),
                format_func=lambda x: collection_options[x],
                help="Select specific collection or search across all"
            )
            
            selected_collection = None if selected_collection_idx == 0 else collections[selected_collection_idx - 1]
            
            # Result limit
            result_limit = st.slider(
                "Maximum Results",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Limit the number of search results"
            )
        
        with col2:
            # Advanced filters
            st.write("**Metadata Filters:**")
            
            # Filter by content type
            filter_type = st.selectbox(
                "Content Type",
                ["Any", "pattern", "decision", "code", "issue"],
                help="Filter by document type if available in metadata"
            )
            
            # Filter by date range (placeholder)
            st.write("*Date range filtering coming in Phase 3*")
    
    # Build filters dictionary
    filters = {}
    if filter_type != "Any":
        filters["type"] = filter_type
    
    # Return search parameters
    return search_query, selected_collection, filters, search_button, result_limit


def render_search_results(results: List[Dict[str, Any]], query: str) -> None:
    """
    Render enhanced search results with multiple view options.
    
    Args:
        results: List of search result dictionaries
        query: Original search query for highlighting
    """
    if not results:
        st.info("No results found. Try different search terms or adjust your filters.")
        return
    
    # Filter out error results
    valid_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]
    
    # Show errors if any
    if error_results:
        with st.expander("⚠️ Search Errors", expanded=False):
            for error in error_results:
                st.error(f"Error: {error.get('error', 'Unknown error')}")
    
    if not valid_results:
        st.warning("All search results contained errors.")
        return
    
    # Results header
    st.subheader(f"🎯 Search Results ({len(valid_results)} found)")
    
    # Results view options
    view_option = st.radio(
        "View Mode:",
        ["📋 List View", "📊 Analytics View", "🎛️ Detailed View"],
        horizontal=True
    )
    
    if view_option == "📋 List View":
        render_list_view(valid_results, query)
    elif view_option == "📊 Analytics View":
        render_analytics_view(valid_results)
    else:
        render_detailed_view(valid_results, query)


def render_list_view(results: List[Dict[str, Any]], query: str) -> None:
    """Render search results in a compact list view."""
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by:",
        ["Relevance", "Collection", "Document ID"],
        key="list_sort"
    )
    
    # Sort results
    if sort_by == "Relevance":
        sorted_results = sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
    elif sort_by == "Collection":
        sorted_results = sorted(results, key=lambda x: x.get('collection', ''))
    else:
        sorted_results = sorted(results, key=lambda x: x.get('id', ''))
    
    # Display results
    for i, result in enumerate(sorted_results):
        with st.container():
            # Result header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{result.get('id', 'Unknown ID')}**")
            
            with col2:
                collection = result.get('collection', 'Unknown')
                st.badge(collection, type="secondary")
            
            with col3:
                relevance = result.get('relevance_score', 0)
                st.write(f"Score: {relevance:.3f}")
            
            # Content preview
            content = result.get('content', '')
            preview = create_content_preview_with_highlights(content, query, max_length=200)
            st.write(preview)
            
            # Metadata summary
            metadata = result.get('metadata', {})
            if metadata:
                metadata_text = " • ".join([f"{k}: {v}" for k, v in list(metadata.items())[:3]])
                st.caption(f"Metadata: {metadata_text}")
            
            # View full document button
            if st.button(f"View Full Document", key=f"view_{i}"):
                with st.expander(f"Full Document: {result.get('id', 'Unknown')}"):
                    st.text_area(
                        "Full Content",
                        content,
                        height=300,
                        disabled=True,
                        key=f"full_content_{i}"
                    )
                    
                    if metadata:
                        st.subheader("Metadata")
                        st.json(metadata)
            
            st.divider()


def render_analytics_view(results: List[Dict[str, Any]]) -> None:
    """Render search results with analytics and visualizations."""
    
    # Collection distribution
    collection_counts = {}
    relevance_scores = []
    
    for result in results:
        collection = result.get('collection', 'Unknown')
        collection_counts[collection] = collection_counts.get(collection, 0) + 1
        relevance_scores.append(result.get('relevance_score', 0))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Collection distribution pie chart
        if len(collection_counts) > 1:
            fig_pie = px.pie(
                values=list(collection_counts.values()),
                names=list(collection_counts.keys()),
                title="Results by Collection"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("All results from single collection")
    
    with col2:
        # Relevance score distribution
        if relevance_scores:
            fig_hist = px.histogram(
                x=relevance_scores,
                nbins=10,
                title="Relevance Score Distribution"
            )
            fig_hist.update_layout(
                xaxis_title="Relevance Score",
                yaxis_title="Number of Documents"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Top results summary
    st.subheader("📊 Results Summary")
    
    # Calculate statistics
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    max_relevance = max(relevance_scores) if relevance_scores else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Relevance", f"{avg_relevance:.3f}")
    
    with col2:
        st.metric("Best Match", f"{max_relevance:.3f}")
    
    with col3:
        st.metric("Collections Searched", len(collection_counts))
    
    # Top 5 results
    st.subheader("🏆 Top 5 Results")
    top_results = sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)[:5]
    
    for i, result in enumerate(top_results, 1):
        with st.container():
            st.write(f"**{i}. {result.get('id', 'Unknown')}** (Score: {result.get('relevance_score', 0):.3f})")
            content = result.get('content', '')
            preview = content[:150] + "..." if len(content) > 150 else content
            st.write(preview)
            st.caption(f"Collection: {result.get('collection', 'Unknown')}")


def render_detailed_view(results: List[Dict[str, Any]], query: str) -> None:
    """Render search results in detailed view with full content."""
    
    # Pagination for detailed view
    results_per_page = 5
    total_pages = (len(results) + results_per_page - 1) // results_per_page
    
    if 'detail_page' not in st.session_state:
        st.session_state.detail_page = 0
    
    # Page navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous") and st.session_state.detail_page > 0:
            st.session_state.detail_page -= 1
            st.rerun()
    
    with col2:
        st.write(f"Page {st.session_state.detail_page + 1} of {total_pages}")
    
    with col3:
        if st.button("➡️ Next") and st.session_state.detail_page < total_pages - 1:
            st.session_state.detail_page += 1
            st.rerun()
    
    # Get results for current page
    start_idx = st.session_state.detail_page * results_per_page
    end_idx = min(start_idx + results_per_page, len(results))
    page_results = results[start_idx:end_idx]
    
    # Display detailed results
    for i, result in enumerate(page_results):
        with st.expander(
            f"📄 {result.get('id', 'Unknown')} "
            f"(Relevance: {result.get('relevance_score', 0):.3f}) "
            f"[{result.get('collection', 'Unknown')}]",
            expanded=True
        ):
            # Content with highlighting
            content = result.get('content', '')
            highlighted_content = create_content_preview_with_highlights(content, query)
            
            st.text_area(
                "Content",
                highlighted_content,
                height=200,
                disabled=True,
                key=f"detailed_content_{start_idx + i}"
            )
            
            # Metadata
            metadata = result.get('metadata', {})
            if metadata:
                st.subheader("Metadata")
                
                # Format metadata nicely
                metadata_cols = st.columns(min(3, len(metadata)))
                for j, (key, value) in enumerate(metadata.items()):
                    with metadata_cols[j % len(metadata_cols)]:
                        st.write(f"**{key}:** {value}")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"📋 Copy ID", key=f"copy_id_{start_idx + i}"):
                    st.write(f"Document ID: `{result.get('id', 'Unknown')}`")
            
            with col2:
                if st.button(f"🔍 Find Similar", key=f"similar_{start_idx + i}"):
                    st.info("Similar document search coming in Phase 3")
            
            with col3:
                if st.button(f"📤 Export", key=f"export_{start_idx + i}"):
                    export_data = {
                        "document": result,
                        "exported_at": datetime.utcnow().isoformat()
                    }
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"document_{result.get('id', 'unknown')}.json",
                        mime="application/json",
                        key=f"download_{start_idx + i}"
                    )


def create_content_preview_with_highlights(content: str, query: str, max_length: int = None) -> str:
    """
    Create content preview with search term highlighting.
    
    Args:
        content: Full content text
        query: Search query to highlight
        max_length: Maximum length of preview
        
    Returns:
        Content with highlighted search terms
    """
    if not content or not query:
        return content[:max_length] if max_length else content
    
    # Simple highlighting (in real app, would use proper text highlighting)
    highlighted = content
    
    # Split query into terms and highlight each
    terms = query.lower().split()
    for term in terms:
        if len(term) >= 3:  # Only highlight meaningful terms
            # Use regex for case-insensitive highlighting
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f"**{term.upper()}**", highlighted)
    
    # Truncate if needed
    if max_length and len(highlighted) > max_length:
        highlighted = highlighted[:max_length] + "..."
    
    return highlighted


def render_search_suggestions(collections: List[str]) -> None:
    """Render search suggestions and tips."""
    
    with st.expander("💡 Search Tips & Examples", expanded=False):
        st.markdown("""
        ### Search Tips
        
        **Effective Search Strategies:**
        - Use specific technical terms: `authentication`, `database`, `API`
        - Search for patterns: `error handling`, `validation pattern`
        - Find decisions: `architecture decision`, `technology choice`
        - Look for solutions: `bug fix`, `performance optimization`
        
        **Example Searches:**
        - `"user authentication implementation"`
        - `"database connection pooling"`
        - `"error handling best practices"`
        - `"API rate limiting"`
        - `"performance optimization"`
        
        **Collection-Specific Searches:**
        - **Patterns**: Search for reusable code templates and designs
        - **Decisions**: Find architectural choices and their rationale
        - **Code Snippets**: Discover useful code examples
        - **Issues**: Explore resolved problems and solutions
        """)
        
        # Quick search buttons
        st.write("**Quick Searches:**")
        
        quick_searches = [
            "authentication",
            "database",
            "error handling", 
            "performance",
            "security",
            "testing"
        ]
        
        cols = st.columns(len(quick_searches))
        for i, search_term in enumerate(quick_searches):
            with cols[i]:
                if st.button(f"🔍 {search_term}", key=f"quick_{i}"):
                    st.session_state.search_query = search_term
                    st.rerun()


def render_no_results_help() -> None:
    """Render helpful suggestions when no results are found."""
    
    st.info("""
    ### No Results Found 
    
    **Try these suggestions:**
    
    1. **Broaden your search terms**
       - Use more general keywords
       - Remove quotes for exact phrase matching
       
    2. **Check your spelling**
       - Verify search terms are spelled correctly
       - Try alternative spellings or synonyms
    
    3. **Search in all collections**
       - Make sure "All Collections" is selected
       - Some content might be in unexpected collections
    
    4. **Use different keywords**
       - Try related technical terms
       - Search for problem descriptions instead of solutions
    
    5. **Check if your knowledge base has content**
       - Go to Overview to see if documents are available
       - Visit Collections to browse available content
    """)


def render_search_page_enhanced(data_access) -> None:
    """
    Render the complete enhanced search page.
    
    Args:
        data_access: Data access instance for searching
    """
    st.title("🔍 Advanced Document Search")
    st.markdown("*Search across your LightRAG knowledge base with powerful filtering*")
    
    try:
        # Get available collections
        overview = data_access.get_collection_overview()
        collections = list(overview.get('collections', {}).keys()) if 'error' not in overview else []
        
        if not collections:
            st.warning("No collections available for searching.")
            return
        
        # Render search interface
        search_query, selected_collection, filters, search_button, result_limit = render_search_interface(collections)
        
        # Search suggestions
        render_search_suggestions(collections)
        
        # Perform search if requested
        if search_button and search_query:
            with st.spinner("🔍 Searching your knowledge base..."):
                results = data_access.search_documents(
                    query=search_query,
                    collection_name=selected_collection,
                    filters=filters if filters else None,
                    limit=result_limit
                )
                
                st.session_state.search_results = results
        
        # Display results if available
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            render_search_results(st.session_state.search_results, search_query)
        elif search_query and search_button:
            render_no_results_help()
    
    except Exception as e:
        st.error(f"Search functionality error: {e}")
        logger.error(f"Enhanced search error: {e}")