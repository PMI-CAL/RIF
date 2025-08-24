"""
Statistics and analytics components for LightRAG visualization.
Detailed metrics and insights about knowledge base content.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime
from collections import Counter
import re

logger = logging.getLogger(__name__)


def render_collection_statistics(data_access) -> None:
    """
    Render detailed statistics for all collections.
    
    Args:
        data_access: Data access instance
    """
    st.subheader("📊 Collection Statistics")
    
    try:
        overview = data_access.get_collection_overview()
        
        if 'error' in overview:
            st.error(f"Error loading statistics: {overview['error']}")
            return
        
        collections = overview.get('collections', {})
        
        if not collections:
            st.warning("No collections available for statistics.")
            return
        
        # Prepare statistics data
        stats_data = []
        for name, stats in collections.items():
            if 'error' not in stats:
                stats_data.append({
                    'Collection': name,
                    'Documents': stats.get('count', 0),
                    'Description': stats.get('description', 'No description'),
                    'Status': 'Active' if stats.get('count', 0) > 0 else 'Empty'
                })
        
        if not stats_data:
            st.warning("No valid collection data for statistics.")
            return
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_docs = sum(item['Documents'] for item in stats_data)
        active_collections = len([item for item in stats_data if item['Status'] == 'Active'])
        avg_docs = total_docs / len(stats_data) if stats_data else 0
        largest_collection = max(stats_data, key=lambda x: x['Documents']) if stats_data else None
        
        with col1:
            st.metric("Total Documents", f"{total_docs:,}")
        
        with col2:
            st.metric("Active Collections", f"{active_collections}/{len(stats_data)}")
        
        with col3:
            st.metric("Avg. Docs/Collection", f"{avg_docs:.1f}")
        
        with col4:
            if largest_collection:
                st.metric("Largest Collection", f"{largest_collection['Collection']} ({largest_collection['Documents']})")
        
        # Visualizations
        render_collection_charts(stats_data)
        
    except Exception as e:
        st.error(f"Failed to load collection statistics: {e}")
        logger.error(f"Collection statistics error: {e}")


def render_collection_charts(stats_data: List[Dict[str, Any]]) -> None:
    """
    Render various charts for collection statistics.
    
    Args:
        stats_data: List of collection statistics
    """
    # Create tabs for different chart types
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📈 Comparison", "🎯 Health"])
    
    with tab1:
        # Document distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig_bar = px.bar(
                stats_data,
                x='Collection',
                y='Documents',
                color='Status',
                title='Documents per Collection',
                color_discrete_map={'Active': '#2E8B57', 'Empty': '#FFA500'}
            )
            fig_bar.update_layout(xaxis_title="Collections", yaxis_title="Document Count")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie chart (only if multiple collections with data)
            active_data = [item for item in stats_data if item['Documents'] > 0]
            if len(active_data) > 1:
                fig_pie = px.pie(
                    active_data,
                    values='Documents',
                    names='Collection',
                    title='Document Distribution (Active Collections)'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Pie chart requires multiple active collections")
    
    with tab2:
        # Collection comparison
        if len(stats_data) > 1:
            # Horizontal bar chart for better readability with long names
            fig_horizontal = px.bar(
                stats_data,
                x='Documents',
                y='Collection',
                orientation='h',
                title='Collection Size Comparison',
                color='Documents',
                color_continuous_scale='viridis'
            )
            fig_horizontal.update_layout(
                xaxis_title="Number of Documents",
                yaxis_title="Collections"
            )
            st.plotly_chart(fig_horizontal, use_container_width=True)
            
            # Statistics table
            st.subheader("Detailed Statistics")
            st.dataframe(stats_data, use_container_width=True, hide_index=True)
        else:
            st.info("Comparison requires multiple collections")
    
    with tab3:
        # Collection health indicators
        render_collection_health_chart(stats_data)


def render_collection_health_chart(stats_data: List[Dict[str, Any]]) -> None:
    """
    Render collection health visualization.
    
    Args:
        stats_data: List of collection statistics
    """
    # Health categories
    health_counts = Counter(item['Status'] for item in stats_data)
    
    # Health pie chart
    fig_health = px.pie(
        values=list(health_counts.values()),
        names=list(health_counts.keys()),
        title='Collection Health Overview',
        color_discrete_map={'Active': '#2E8B57', 'Empty': '#FFA500'}
    )
    st.plotly_chart(fig_health, use_container_width=True)
    
    # Health details
    st.subheader("Health Status Details")
    
    for item in stats_data:
        col1, col2, col3 = st.columns([2, 1, 3])
        
        with col1:
            st.write(f"**{item['Collection']}**")
        
        with col2:
            if item['Status'] == 'Active':
                st.success("🟢 Active")
            else:
                st.warning("🟡 Empty")
        
        with col3:
            if item['Status'] == 'Active':
                st.write(f"{item['Documents']:,} documents")
            else:
                st.write("No documents - consider populating or removing")


def render_document_analytics(data_access) -> None:
    """
    Render detailed document analytics.
    
    Args:
        data_access: Data access instance
    """
    st.subheader("📄 Document Analytics")
    
    # Collection selector for detailed analysis
    overview = data_access.get_collection_overview()
    collections = list(overview.get('collections', {}).keys()) if 'error' not in overview else []
    
    if not collections:
        st.warning("No collections available for document analytics.")
        return
    
    selected_collection = st.selectbox(
        "Select collection for detailed analysis:",
        collections,
        key="analytics_collection"
    )
    
    if selected_collection:
        with st.spinner("Analyzing documents..."):
            analytics_data = analyze_collection_documents(data_access, selected_collection)
            
            if analytics_data:
                render_document_analytics_charts(analytics_data, selected_collection)
            else:
                st.warning(f"No data available for analysis in {selected_collection}")


def analyze_collection_documents(data_access, collection_name: str) -> Optional[Dict[str, Any]]:
    """
    Analyze documents in a collection for statistics.
    
    Args:
        data_access: Data access instance
        collection_name: Name of collection to analyze
        
    Returns:
        Analytics data dictionary or None
    """
    try:
        # Get sample of documents for analysis
        page_data = data_access.get_documents_paginated(
            collection_name=collection_name,
            page=0,
            page_size=100  # Analyze first 100 documents
        )
        
        if 'error' in page_data:
            return None
        
        documents = page_data.get('documents', [])
        
        if not documents:
            return None
        
        # Analyze documents
        content_lengths = []
        metadata_fields = set()
        word_counts = []
        content_types = Counter()
        
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Content analysis
            content_lengths.append(len(content))
            word_counts.append(len(content.split()) if content else 0)
            
            # Metadata analysis
            if metadata:
                metadata_fields.update(metadata.keys())
                
                # Content type analysis
                doc_type = metadata.get('type', 'unknown')
                content_types[doc_type] += 1
        
        analytics = {
            'total_documents': len(documents),
            'total_available': page_data.get('pagination', {}).get('total_count', len(documents)),
            'content_lengths': content_lengths,
            'word_counts': word_counts,
            'metadata_fields': list(metadata_fields),
            'content_types': dict(content_types),
            'sample_size': len(documents)
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Document analysis error: {e}")
        return None


def render_document_analytics_charts(analytics: Dict[str, Any], collection_name: str) -> None:
    """
    Render charts for document analytics.
    
    Args:
        analytics: Analytics data dictionary
        collection_name: Name of analyzed collection
    """
    # Sample size indicator
    if analytics['sample_size'] < analytics['total_available']:
        st.info(f"📊 Analysis based on sample of {analytics['sample_size']} documents "
                f"out of {analytics['total_available']} total documents")
    
    # Document metrics
    col1, col2, col3, col4 = st.columns(4)
    
    content_lengths = analytics['content_lengths']
    word_counts = analytics['word_counts']
    
    with col1:
        avg_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
        st.metric("Avg. Content Length", f"{avg_length:.0f} chars")
    
    with col2:
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        st.metric("Avg. Word Count", f"{avg_words:.0f} words")
    
    with col3:
        st.metric("Metadata Fields", len(analytics['metadata_fields']))
    
    with col4:
        st.metric("Content Types", len(analytics['content_types']))
    
    # Charts
    tab1, tab2, tab3 = st.tabs(["📏 Content Analysis", "🏷️ Metadata Analysis", "📑 Content Types"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Content length distribution
            fig_length = px.histogram(
                x=content_lengths,
                nbins=20,
                title="Content Length Distribution",
                labels={'x': 'Content Length (characters)', 'y': 'Number of Documents'}
            )
            st.plotly_chart(fig_length, use_container_width=True)
        
        with col2:
            # Word count distribution
            fig_words = px.histogram(
                x=word_counts,
                nbins=20,
                title="Word Count Distribution",
                labels={'x': 'Word Count', 'y': 'Number of Documents'}
            )
            st.plotly_chart(fig_words, use_container_width=True)
        
        # Content length statistics
        if content_lengths:
            st.subheader("Content Length Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Minimum", f"{min(content_lengths)} chars")
            
            with col2:
                st.metric("Maximum", f"{max(content_lengths)} chars")
            
            with col3:
                median_length = sorted(content_lengths)[len(content_lengths) // 2]
                st.metric("Median", f"{median_length} chars")
            
            with col4:
                # Calculate standard deviation
                import statistics
                std_dev = statistics.stdev(content_lengths) if len(content_lengths) > 1 else 0
                st.metric("Std. Deviation", f"{std_dev:.0f}")
    
    with tab2:
        # Metadata field analysis
        if analytics['metadata_fields']:
            st.subheader("Available Metadata Fields")
            
            # Display metadata fields as tags
            for field in analytics['metadata_fields']:
                st.code(field)
        else:
            st.info("No metadata fields found in analyzed documents")
    
    with tab3:
        # Content types analysis
        content_types = analytics['content_types']
        
        if content_types and len(content_types) > 1:
            # Content types pie chart
            fig_types = px.pie(
                values=list(content_types.values()),
                names=list(content_types.keys()),
                title="Content Types Distribution"
            )
            st.plotly_chart(fig_types, use_container_width=True)
            
            # Content types table
            st.subheader("Content Types Summary")
            types_data = [
                {'Type': type_name, 'Count': count, 'Percentage': f"{(count/sum(content_types.values()))*100:.1f}%"}
                for type_name, count in content_types.items()
            ]
            st.dataframe(types_data, use_container_width=True, hide_index=True)
        else:
            st.info("No content type diversity found in metadata")


def render_knowledge_insights(data_access) -> None:
    """
    Render knowledge base insights and recommendations.
    
    Args:
        data_access: Data access instance
    """
    st.subheader("💡 Knowledge Base Insights")
    
    try:
        overview = data_access.get_collection_overview()
        
        if 'error' in overview:
            st.error(f"Error loading insights: {overview['error']}")
            return
        
        collections = overview.get('collections', {})
        total_docs = overview.get('total_documents', 0)
        total_collections = overview.get('total_collections', 0)
        
        # Generate insights
        insights = generate_knowledge_insights(collections, total_docs, total_collections)
        
        # Display insights in cards
        for insight in insights:
            with st.container():
                if insight['type'] == 'success':
                    st.success(f"✅ **{insight['title']}**\n\n{insight['message']}")
                elif insight['type'] == 'warning':
                    st.warning(f"⚠️ **{insight['title']}**\n\n{insight['message']}")
                elif insight['type'] == 'info':
                    st.info(f"ℹ️ **{insight['title']}**\n\n{insight['message']}")
                else:
                    st.write(f"**{insight['title']}**\n\n{insight['message']}")
        
        # Recommendations
        if insights:
            st.subheader("🎯 Recommendations")
            
            recommendations = generate_recommendations(collections, total_docs, total_collections)
            
            for rec in recommendations:
                st.write(f"• **{rec['action']}**: {rec['description']}")
    
    except Exception as e:
        st.error(f"Failed to generate insights: {e}")
        logger.error(f"Knowledge insights error: {e}")


def generate_knowledge_insights(collections: Dict[str, Any], total_docs: int, total_collections: int) -> List[Dict[str, Any]]:
    """
    Generate insights based on knowledge base analysis.
    
    Args:
        collections: Collections data
        total_docs: Total number of documents
        total_collections: Total number of collections
        
    Returns:
        List of insight dictionaries
    """
    insights = []
    
    # Analyze collection balance
    doc_counts = [stats.get('count', 0) for stats in collections.values() if 'count' in stats]
    active_collections = [name for name, stats in collections.items() if stats.get('count', 0) > 0]
    empty_collections = [name for name, stats in collections.items() if stats.get('count', 0) == 0]
    
    # Total documents insight
    if total_docs == 0:
        insights.append({
            'type': 'warning',
            'title': 'Empty Knowledge Base',
            'message': 'Your knowledge base contains no documents. Start by using RIF agents to populate it with patterns, decisions, and code snippets.'
        })
    elif total_docs < 50:
        insights.append({
            'type': 'info',
            'title': 'Growing Knowledge Base',
            'message': f'You have {total_docs} documents. Continue using RIF agents to build a comprehensive knowledge repository.'
        })
    elif total_docs < 500:
        insights.append({
            'type': 'success',
            'title': 'Healthy Knowledge Base',
            'message': f'Good progress with {total_docs} documents! Your knowledge base is developing well.'
        })
    else:
        insights.append({
            'type': 'success',
            'title': 'Rich Knowledge Repository',
            'message': f'Excellent! You have {total_docs} documents. This is a substantial knowledge base that can provide valuable insights.'
        })
    
    # Collection balance insight
    if len(empty_collections) > 0:
        insights.append({
            'type': 'info',
            'title': f'Unused Collections ({len(empty_collections)})',
            'message': f'Collections {", ".join(empty_collections)} are empty. Consider populating them or focusing on active collections.'
        })
    
    # Collection distribution insight
    if doc_counts and len(doc_counts) > 1:
        max_docs = max(doc_counts)
        min_docs = min([count for count in doc_counts if count > 0])
        
        if max_docs > min_docs * 10:  # Very unbalanced
            insights.append({
                'type': 'warning',
                'title': 'Unbalanced Collections',
                'message': 'Some collections have significantly more documents than others. Consider balancing content distribution.'
            })
        elif max_docs > min_docs * 3:  # Somewhat unbalanced
            insights.append({
                'type': 'info',
                'title': 'Collection Imbalance',
                'message': 'Document distribution across collections is somewhat uneven. This is normal but monitor for extreme imbalances.'
            })
    
    return insights


def generate_recommendations(collections: Dict[str, Any], total_docs: int, total_collections: int) -> List[Dict[str, Any]]:
    """
    Generate actionable recommendations.
    
    Args:
        collections: Collections data
        total_docs: Total number of documents
        total_collections: Total number of collections
        
    Returns:
        List of recommendation dictionaries
    """
    recommendations = []
    
    active_collections = [name for name, stats in collections.items() if stats.get('count', 0) > 0]
    empty_collections = [name for name, stats in collections.items() if stats.get('count', 0) == 0]
    
    if total_docs == 0:
        recommendations.append({
            'action': 'Start Knowledge Collection',
            'description': 'Begin by documenting your first patterns and decisions using RIF agents'
        })
    
    if len(empty_collections) > 0:
        recommendations.append({
            'action': 'Populate Empty Collections',
            'description': f'Consider adding content to {", ".join(empty_collections[:2])} collections'
        })
    
    if total_docs > 100:
        recommendations.append({
            'action': 'Use Advanced Analytics',
            'description': 'With substantial content, explore document relationships and pattern analysis'
        })
    
    if len(active_collections) == 1:
        recommendations.append({
            'action': 'Diversify Knowledge Types',
            'description': 'Expand beyond single collection to capture different types of knowledge'
        })
    
    recommendations.append({
        'action': 'Regular Review',
        'description': 'Periodically review and update your knowledge base to keep it current and valuable'
    })
    
    return recommendations


def render_statistics_page(data_access) -> None:
    """
    Render the complete statistics and analytics page.
    
    Args:
        data_access: Data access instance
    """
    st.title("📊 Knowledge Base Analytics")
    st.markdown("*Detailed statistics and insights about your LightRAG data*")
    
    # Main sections
    render_collection_statistics(data_access)
    st.divider()
    
    render_document_analytics(data_access)
    st.divider()
    
    render_knowledge_insights(data_access)