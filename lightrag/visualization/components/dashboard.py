"""
Dashboard components for LightRAG visualization.
Enhanced collection overview and statistics displays.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def render_collection_metrics(overview: Dict[str, Any]) -> None:
    """
    Render collection metrics in an enhanced dashboard format.
    
    Args:
        overview: Collection overview data from data access layer
    """
    if 'error' in overview:
        st.error(f"Error loading metrics: {overview['error']}")
        return
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📚 Total Collections",
            value=overview.get('total_collections', 0),
            help="Number of knowledge collections in the database"
        )
    
    with col2:
        st.metric(
            label="📄 Total Documents", 
            value=f"{overview.get('total_documents', 0):,}",
            help="Total number of documents across all collections"
        )
    
    with col3:
        # Calculate average documents per collection
        total_colls = overview.get('total_collections', 0)
        total_docs = overview.get('total_documents', 0)
        avg_docs = round(total_docs / total_colls) if total_colls > 0 else 0
        st.metric(
            label="📊 Avg. Docs/Collection",
            value=avg_docs,
            help="Average number of documents per collection"
        )
    
    with col4:
        # Database status indicator
        db_path = overview.get('database_path', '')
        status = "🟢 Connected" if db_path else "🔴 Disconnected"
        st.metric(
            label="🗄️ Database Status",
            value=status,
            help=f"Database path: {db_path}" if db_path else "No database connection"
        )


def render_collection_overview_chart(collections: Dict[str, Any]) -> None:
    """
    Render an enhanced collection overview chart.
    
    Args:
        collections: Collections data from overview
    """
    if not collections:
        st.info("No collections data available for visualization.")
        return
    
    # Prepare data for visualization
    collection_data = []
    for name, stats in collections.items():
        if 'error' not in stats and 'count' in stats:
            collection_data.append({
                'name': name,
                'count': stats['count'],
                'description': stats.get('description', 'No description')
            })
    
    if not collection_data:
        st.warning("No valid collection data found for visualization.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart", "📋 Data Table"])
    
    with tab1:
        # Enhanced bar chart
        fig_bar = px.bar(
            collection_data,
            x='name',
            y='count',
            title='Documents per Collection',
            labels={'count': 'Number of Documents', 'name': 'Collection'},
            color='count',
            color_continuous_scale='viridis',
            hover_data=['description']
        )
        
        fig_bar.update_layout(
            showlegend=False,
            xaxis_title="Collections",
            yaxis_title="Document Count",
            title_x=0.5
        )
        
        # Add value labels on bars
        fig_bar.update_traces(texttemplate='%{y}', textposition='outside')
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        # Enhanced pie chart
        if len(collection_data) > 1:  # Only show pie chart if multiple collections
            fig_pie = px.pie(
                collection_data,
                values='count',
                names='name',
                title='Document Distribution by Collection',
                hover_data=['description']
            )
            
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(title_x=0.5)
            
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Pie chart requires multiple collections with documents.")
    
    with tab3:
        # Enhanced data table
        st.subheader("Collection Details")
        
        # Create enhanced table data
        table_data = []
        for item in collection_data:
            table_data.append({
                'Collection': item['name'],
                'Documents': f"{item['count']:,}",
                'Description': item['description'][:100] + "..." if len(item['description']) > 100 else item['description']
            })
        
        if table_data:
            # Use st.dataframe for better interactivity
            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No collection data available for table view.")


def render_collection_health_indicators(collections: Dict[str, Any]) -> None:
    """
    Render health indicators for collections.
    
    Args:
        collections: Collections data from overview
    """
    st.subheader("🏥 Collection Health")
    
    health_data = []
    for name, stats in collections.items():
        if 'error' in stats:
            status = "🔴 Error"
            health = "error"
            message = stats['error']
        elif stats.get('count', 0) == 0:
            status = "🟡 Empty"
            health = "warning"
            message = "Collection contains no documents"
        else:
            status = "🟢 Healthy"
            health = "success"
            message = f"{stats['count']:,} documents available"
        
        health_data.append({
            'Collection': name,
            'Status': status,
            'Health': health,
            'Message': message
        })
    
    # Display health indicators
    for item in health_data:
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 4])
            
            with col1:
                st.write(f"**{item['Collection']}**")
            
            with col2:
                st.write(item['Status'])
            
            with col3:
                if item['Health'] == 'error':
                    st.error(item['Message'])
                elif item['Health'] == 'warning':
                    st.warning(item['Message'])
                else:
                    st.success(item['Message'])


def render_database_info(overview: Dict[str, Any]) -> None:
    """
    Render detailed database information.
    
    Args:
        overview: Overview data containing database info
    """
    with st.expander("🗄️ Database Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Database Path:**")
            st.code(overview.get('database_path', 'Unknown'))
            
            st.write("**Last Updated:**")
            last_updated = overview.get('last_updated', 'Unknown')
            if last_updated != 'Unknown':
                try:
                    # Format timestamp if it's a valid ISO string
                    dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
                    st.code(formatted_time)
                except:
                    st.code(last_updated)
            else:
                st.code(last_updated)
        
        with col2:
            st.write("**Collections Available:**")
            for name in overview.get('collections', {}).keys():
                st.write(f"• {name}")
            
            if not overview.get('collections'):
                st.write("No collections found")


def render_quick_actions(collections: Dict[str, Any]) -> None:
    """
    Render quick action buttons for common operations.
    
    Args:
        collections: Collections data for action context
    """
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Search All", help="Search across all collections"):
            st.session_state.navigation = "Search"
            st.rerun()
    
    with col2:
        if st.button("📚 Browse Collections", help="Browse collection details"):
            st.session_state.navigation = "Collections"
            st.rerun()
    
    with col3:
        if st.button("📄 Document Browser", help="Browse individual documents"):
            st.session_state.navigation = "Document Browser"
            st.rerun()
    
    with col4:
        if st.button("📊 Analytics", help="View analytics and insights"):
            st.session_state.navigation = "Analytics"
            st.rerun()


def render_collection_comparison(collections: Dict[str, Any]) -> None:
    """
    Render a comparison view of collections.
    
    Args:
        collections: Collections data for comparison
    """
    if not collections or len(collections) < 2:
        return
    
    st.subheader("📈 Collection Comparison")
    
    # Prepare comparison data
    comparison_data = []
    for name, stats in collections.items():
        if 'count' in stats:
            comparison_data.append({
                'Collection': name,
                'Documents': stats['count'],
                'Status': 'Active' if stats['count'] > 0 else 'Empty'
            })
    
    if len(comparison_data) < 2:
        st.info("Need at least 2 collections for comparison.")
        return
    
    # Sort by document count
    comparison_data.sort(key=lambda x: x['Documents'], reverse=True)
    
    # Create comparison chart
    fig = px.bar(
        comparison_data,
        x='Collection',
        y='Documents',
        color='Status',
        title='Collection Size Comparison',
        color_discrete_map={'Active': '#2E8B57', 'Empty': '#FFA500'}
    )
    
    fig.update_layout(
        xaxis_title="Collections",
        yaxis_title="Number of Documents",
        title_x=0.5,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top collections
    if comparison_data:
        largest = comparison_data[0]
        smallest = [c for c in comparison_data if c['Documents'] > 0]
        smallest = smallest[-1] if smallest else None
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "🏆 Largest Collection",
                f"{largest['Collection']}: {largest['Documents']:,} docs"
            )
        
        with col2:
            if smallest and smallest != largest:
                st.metric(
                    "📝 Smallest Active Collection",
                    f"{smallest['Collection']}: {smallest['Documents']:,} docs"
                )


def render_recent_activity(overview: Dict[str, Any]) -> None:
    """
    Render recent activity information (placeholder for Phase 3).
    
    Args:
        overview: Overview data
    """
    st.subheader("🕒 Recent Activity")
    
    # This is a placeholder for Phase 3 implementation
    st.info("""
    📈 **Recent Activity Timeline** (Coming in Phase 3)
    
    This section will show:
    • Recent document additions
    • Collection growth trends
    • Agent activity patterns
    • Knowledge base updates
    """)
    
    # Show last database update as basic activity info
    last_updated = overview.get('last_updated', 'Unknown')
    if last_updated != 'Unknown':
        st.write(f"**Last Database Update:** {last_updated}")


def render_collection_insights(collections: Dict[str, Any]) -> None:
    """
    Render insights and recommendations based on collection data.
    
    Args:
        collections: Collections data for analysis
    """
    st.subheader("💡 Insights & Recommendations")
    
    # Analyze collections
    total_docs = sum(stats.get('count', 0) for stats in collections.values() if 'count' in stats)
    active_collections = [name for name, stats in collections.items() if stats.get('count', 0) > 0]
    empty_collections = [name for name, stats in collections.items() if stats.get('count', 0) == 0]
    
    insights = []
    
    if total_docs == 0:
        insights.append({
            'type': 'warning',
            'title': 'No Documents Found',
            'message': 'Your knowledge base appears to be empty. Start by adding some documents to see insights.',
            'action': 'Use RIF agents to populate the knowledge base with patterns and decisions.'
        })
    elif len(empty_collections) > 0:
        insights.append({
            'type': 'info',
            'title': f'{len(empty_collections)} Empty Collections',
            'message': f'Collections {", ".join(empty_collections)} are empty.',
            'action': 'Consider using these collections for new knowledge types or remove if not needed.'
        })
    
    if len(active_collections) == 1:
        insights.append({
            'type': 'info',
            'title': 'Single Active Collection',
            'message': f'Only {active_collections[0]} contains documents.',
            'action': 'Consider diversifying your knowledge base by adding content to other collections.'
        })
    
    if total_docs > 1000:
        insights.append({
            'type': 'success',
            'title': 'Rich Knowledge Base',
            'message': f'Great! You have {total_docs:,} documents across your collections.',
            'action': 'Use advanced analytics to find patterns and relationships in your knowledge.'
        })
    
    # Display insights
    for insight in insights:
        if insight['type'] == 'warning':
            with st.container():
                st.warning(f"⚠️ **{insight['title']}**\n\n{insight['message']}\n\n💡 *{insight['action']}*")
        elif insight['type'] == 'info':
            with st.container():
                st.info(f"ℹ️ **{insight['title']}**\n\n{insight['message']}\n\n💡 *{insight['action']}*")
        elif insight['type'] == 'success':
            with st.container():
                st.success(f"✅ **{insight['title']}**\n\n{insight['message']}\n\n💡 *{insight['action']}*")
    
    if not insights:
        st.success("✅ **Knowledge Base Looks Good!**\n\nYour collections are well-balanced and active.")


def render_enhanced_overview_dashboard(overview: Dict[str, Any]) -> None:
    """
    Render the complete enhanced overview dashboard.
    
    Args:
        overview: Complete overview data from data access layer
    """
    # Header
    st.title("🧠 LightRAG Knowledge Base Dashboard")
    st.markdown("*Comprehensive overview of your intelligent knowledge collections*")
    st.divider()
    
    # Main metrics
    render_collection_metrics(overview)
    st.divider()
    
    # Collection visualizations
    collections = overview.get('collections', {})
    if collections:
        render_collection_overview_chart(collections)
        st.divider()
        
        # Two column layout for additional information
        col1, col2 = st.columns(2)
        
        with col1:
            render_collection_health_indicators(collections)
        
        with col2:
            render_collection_comparison(collections)
        
        st.divider()
        
        # Quick actions
        render_quick_actions(collections)
        st.divider()
        
        # Insights and recommendations
        render_collection_insights(collections)
        st.divider()
        
        # Recent activity (placeholder)
        render_recent_activity(overview)
        
    else:
        st.warning("No collection data available. Please check your LightRAG setup.")
    
    # Database information
    render_database_info(overview)