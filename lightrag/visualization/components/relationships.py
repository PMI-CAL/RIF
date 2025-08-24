"""
LightRAG Relationship Visualization Component
Handles document relationship mapping, network graphs, and connection analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json

logger = logging.getLogger(__name__)

def render_relationships_page():
    """Render the main relationships analysis page."""
    st.header("📊 Document Relationships & Network Analysis")
    
    if 'data_access' not in st.session_state or not st.session_state.app_initialized:
        st.error("Data access not initialized. Please return to the Overview page.")
        return
    
    data_access = st.session_state.data_access
    
    # Page tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Network Graph", 
        "📈 Vector Space", 
        "🔗 Relationship Matrix",
        "🎯 Similarity Clusters"
    ])
    
    with tab1:
        render_network_graph_section(data_access)
    
    with tab2:
        render_vector_space_section(data_access)
    
    with tab3:
        render_relationship_matrix_section(data_access)
    
    with tab4:
        render_similarity_clusters_section(data_access)

def render_network_graph_section(data_access):
    """Render interactive network graph visualization."""
    st.subheader("🌐 Document Network Graph")
    
    # Collection selection
    collections = data_access.get_available_collections()
    if not collections:
        st.warning("No collections found in the database.")
        return
    
    selected_collection = st.selectbox(
        "Select Collection for Network Analysis",
        collections,
        key="network_collection"
    )
    
    # Network parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold", 
            0.1, 0.9, 0.5, 0.05,
            help="Minimum similarity score to create edges"
        )
    
    with col2:
        max_nodes = st.slider(
            "Maximum Nodes", 
            10, 500, 100, 10,
            help="Maximum number of nodes to display"
        )
    
    with col3:
        layout_type = st.selectbox(
            "Layout Algorithm",
            ["spring", "circular", "kamada_kawai", "shell"],
            help="Network layout algorithm"
        )
    
    if st.button("Generate Network Graph", key="generate_network"):
        with st.spinner("Generating network graph..."):
            try:
                network_data = generate_document_network(
                    data_access, selected_collection, 
                    similarity_threshold, max_nodes
                )
                
                if network_data and network_data['nodes']:
                    fig = create_interactive_network_plot(
                        network_data, layout_type
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Network statistics
                    render_network_statistics(network_data)
                else:
                    st.warning("No network connections found with current parameters.")
                    
            except Exception as e:
                st.error(f"Failed to generate network graph: {e}")
                logger.error(f"Network generation error: {e}")

def render_vector_space_section(data_access):
    """Render vector space visualization with clustering."""
    st.subheader("📈 Vector Space Visualization")
    
    collections = data_access.get_available_collections()
    if not collections:
        st.warning("No collections found in the database.")
        return
    
    selected_collection = st.selectbox(
        "Select Collection for Vector Analysis",
        collections,
        key="vector_collection"
    )
    
    # Vector space parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        n_components = st.selectbox(
            "Visualization Dimensions",
            [2, 3],
            help="2D or 3D visualization"
        )
    
    with col2:
        n_clusters = st.slider(
            "Number of Clusters",
            2, 20, 5,
            help="Number of clusters for K-means"
        )
    
    with col3:
        sample_size = st.slider(
            "Sample Size",
            50, 1000, 200, 50,
            help="Number of documents to visualize"
        )
    
    if st.button("Generate Vector Space", key="generate_vectors"):
        with st.spinner("Generating vector space visualization..."):
            try:
                vector_data = generate_vector_space_data(
                    data_access, selected_collection, 
                    n_components, n_clusters, sample_size
                )
                
                if vector_data:
                    if n_components == 2:
                        fig = create_2d_vector_plot(vector_data)
                    else:
                        fig = create_3d_vector_plot(vector_data)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster analysis
                    render_cluster_analysis(vector_data)
                else:
                    st.warning("Insufficient vector data for visualization.")
                    
            except Exception as e:
                st.error(f"Failed to generate vector space: {e}")
                logger.error(f"Vector space error: {e}")

def render_relationship_matrix_section(data_access):
    """Render document relationship matrix heatmap."""
    st.subheader("🔗 Document Relationship Matrix")
    
    collections = data_access.get_available_collections()
    if not collections:
        st.warning("No collections found in the database.")
        return
    
    selected_collection = st.selectbox(
        "Select Collection for Matrix Analysis",
        collections,
        key="matrix_collection"
    )
    
    # Matrix parameters
    col1, col2 = st.columns(2)
    with col1:
        matrix_size = st.slider(
            "Matrix Size (docs)",
            10, 100, 30,
            help="Number of documents for matrix"
        )
    
    with col2:
        similarity_metric = st.selectbox(
            "Similarity Metric",
            ["cosine", "euclidean", "manhattan"],
            help="Distance metric for similarity calculation"
        )
    
    if st.button("Generate Relationship Matrix", key="generate_matrix"):
        with st.spinner("Computing relationship matrix..."):
            try:
                matrix_data = generate_relationship_matrix(
                    data_access, selected_collection, 
                    matrix_size, similarity_metric
                )
                
                if matrix_data:
                    fig = create_relationship_heatmap(matrix_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Matrix statistics
                    render_matrix_statistics(matrix_data)
                else:
                    st.warning("Insufficient data for relationship matrix.")
                    
            except Exception as e:
                st.error(f"Failed to generate relationship matrix: {e}")
                logger.error(f"Matrix generation error: {e}")

def render_similarity_clusters_section(data_access):
    """Render similarity-based document clustering."""
    st.subheader("🎯 Document Similarity Clusters")
    
    collections = data_access.get_available_collections()
    if not collections:
        st.warning("No collections found in the database.")
        return
    
    selected_collection = st.selectbox(
        "Select Collection for Clustering",
        collections,
        key="cluster_collection"
    )
    
    # Clustering parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        clustering_method = st.selectbox(
            "Clustering Method",
            ["kmeans", "hierarchical", "dbscan"],
            help="Clustering algorithm"
        )
    
    with col2:
        n_clusters_param = st.slider(
            "Target Clusters",
            2, 15, 5,
            help="Number of clusters (for K-means)"
        )
    
    with col3:
        min_cluster_size = st.slider(
            "Min Cluster Size",
            2, 20, 5,
            help="Minimum documents per cluster"
        )
    
    if st.button("Perform Clustering Analysis", key="perform_clustering"):
        with st.spinner("Performing clustering analysis..."):
            try:
                cluster_data = perform_document_clustering(
                    data_access, selected_collection,
                    clustering_method, n_clusters_param, min_cluster_size
                )
                
                if cluster_data:
                    # Cluster visualization
                    fig = create_cluster_visualization(cluster_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster details
                    render_cluster_details(cluster_data)
                else:
                    st.warning("Clustering analysis failed or returned no results.")
                    
            except Exception as e:
                st.error(f"Failed to perform clustering: {e}")
                logger.error(f"Clustering error: {e}")

def generate_document_network(data_access, collection: str, threshold: float, max_nodes: int) -> Optional[Dict]:
    """Generate network graph data from document similarities."""
    try:
        # Get documents with embeddings
        documents = data_access.get_documents_with_embeddings(collection, limit=max_nodes)
        if not documents or len(documents) < 2:
            return None
        
        # Calculate similarity matrix
        embeddings = np.array([doc.get('embedding', []) for doc in documents])
        if embeddings.size == 0 or len(embeddings[0]) == 0:
            return None
        
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i, doc in enumerate(documents):
            G.add_node(i, 
                title=doc.get('title', f'Document {i}')[:50],
                content=doc.get('content', '')[:200],
                id=doc.get('id', f'doc_{i}')
            )
        
        # Add edges based on similarity threshold
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i][j]
                if similarity > threshold:
                    G.add_edge(i, j, weight=similarity)
        
        # Extract node and edge data
        nodes = []
        edges = []
        
        for node_id, node_data in G.nodes(data=True):
            nodes.append({
                'id': node_id,
                'title': node_data['title'],
                'content': node_data['content'],
                'degree': G.degree(node_id)
            })
        
        for edge in G.edges(data=True):
            edges.append({
                'source': edge[0],
                'target': edge[1],
                'weight': edge[2]['weight']
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'graph': G
        }
        
    except Exception as e:
        logger.error(f"Network generation error: {e}")
        return None

def create_interactive_network_plot(network_data: Dict, layout_type: str) -> go.Figure:
    """Create interactive network plot using Plotly."""
    G = network_data['graph']
    nodes = network_data['nodes']
    edges = network_data['edges']
    
    # Calculate layout positions
    if layout_type == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:  # shell
        pos = nx.shell_layout(G)
    
    # Extract coordinates
    x_nodes = [pos[node['id']][0] for node in nodes]
    y_nodes = [pos[node['id']][1] for node in nodes]
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in edges:
        x0, y0 = pos[edge['source']]
        x1, y1 = pos[edge['target']]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge['weight'])
    
    # Create the figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Connections'
    ))
    
    # Add nodes
    node_degrees = [node['degree'] for node in nodes]
    node_titles = [node['title'] for node in nodes]
    node_contents = [node['content'] for node in nodes]
    
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        hovertemplate='<b>%{text}</b><br>Connections: %{marker.size}<br>%{customdata}<extra></extra>',
        text=node_titles,
        textposition="middle center",
        customdata=node_contents,
        marker=dict(
            size=[max(10, min(30, degree * 3)) for degree in node_degrees],
            color=node_degrees,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Node Degree")
        ),
        name='Documents'
    ))
    
    fig.update_layout(
        title="Document Relationship Network",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Larger nodes = more connections",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color="gray", size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

def generate_vector_space_data(data_access, collection: str, n_components: int, 
                             n_clusters: int, sample_size: int) -> Optional[Dict]:
    """Generate vector space visualization data."""
    try:
        # Get documents with embeddings
        documents = data_access.get_documents_with_embeddings(collection, limit=sample_size)
        if not documents or len(documents) < n_clusters:
            return None
        
        embeddings = np.array([doc.get('embedding', []) for doc in documents])
        if embeddings.size == 0 or len(embeddings[0]) == 0:
            return None
        
        # Dimensionality reduction
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Prepare data
        titles = [doc.get('title', f'Document {i}')[:50] for i, doc in enumerate(documents)]
        contents = [doc.get('content', '')[:200] for doc in documents]
        
        return {
            'embeddings': reduced_embeddings,
            'clusters': cluster_labels,
            'titles': titles,
            'contents': contents,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'n_components': n_components
        }
        
    except Exception as e:
        logger.error(f"Vector space generation error: {e}")
        return None

def create_2d_vector_plot(vector_data: Dict) -> go.Figure:
    """Create 2D vector space plot."""
    embeddings = vector_data['embeddings']
    clusters = vector_data['clusters']
    titles = vector_data['titles']
    contents = vector_data['contents']
    
    fig = go.Figure()
    
    # Create scatter plot with cluster colors
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        fig.add_trace(go.Scatter(
            x=embeddings[mask, 0],
            y=embeddings[mask, 1],
            mode='markers',
            name=f'Cluster {cluster_id}',
            text=[titles[i] for i in range(len(titles)) if mask[i]],
            customdata=[contents[i] for i in range(len(contents)) if mask[i]],
            hovertemplate='<b>%{text}</b><br>Cluster: %{fullData.name}<br>%{customdata}<extra></extra>',
            marker=dict(size=8, opacity=0.7)
        ))
    
    explained_var = vector_data['pca_explained_variance']
    fig.update_layout(
        title=f"2D Vector Space Visualization<br><sub>PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%} variance explained</sub>",
        xaxis_title=f"Principal Component 1 ({explained_var[0]:.1%})",
        yaxis_title=f"Principal Component 2 ({explained_var[1]:.1%})",
        height=600,
        showlegend=True
    )
    
    return fig

def create_3d_vector_plot(vector_data: Dict) -> go.Figure:
    """Create 3D vector space plot."""
    embeddings = vector_data['embeddings']
    clusters = vector_data['clusters']
    titles = vector_data['titles']
    contents = vector_data['contents']
    
    fig = go.Figure()
    
    # Create 3D scatter plot with cluster colors
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        fig.add_trace(go.Scatter3d(
            x=embeddings[mask, 0],
            y=embeddings[mask, 1],
            z=embeddings[mask, 2],
            mode='markers',
            name=f'Cluster {cluster_id}',
            text=[titles[i] for i in range(len(titles)) if mask[i]],
            customdata=[contents[i] for i in range(len(contents)) if mask[i]],
            hovertemplate='<b>%{text}</b><br>Cluster: %{fullData.name}<br>%{customdata}<extra></extra>',
            marker=dict(size=6, opacity=0.7)
        ))
    
    explained_var = vector_data['pca_explained_variance']
    fig.update_layout(
        title=f"3D Vector Space Visualization<br><sub>PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%}, PC3: {explained_var[2]:.1%} variance explained</sub>",
        scene=dict(
            xaxis_title=f"Principal Component 1 ({explained_var[0]:.1%})",
            yaxis_title=f"Principal Component 2 ({explained_var[1]:.1%})",
            zaxis_title=f"Principal Component 3 ({explained_var[2]:.1%})"
        ),
        height=700,
        showlegend=True
    )
    
    return fig

def generate_relationship_matrix(data_access, collection: str, matrix_size: int, 
                               similarity_metric: str) -> Optional[Dict]:
    """Generate document relationship matrix."""
    try:
        documents = data_access.get_documents_with_embeddings(collection, limit=matrix_size)
        if not documents or len(documents) < 2:
            return None
        
        embeddings = np.array([doc.get('embedding', []) for doc in documents])
        if embeddings.size == 0:
            return None
        
        # Calculate similarity matrix
        if similarity_metric == "cosine":
            similarity_matrix = cosine_similarity(embeddings)
        else:
            from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
            if similarity_metric == "euclidean":
                distance_matrix = euclidean_distances(embeddings)
                similarity_matrix = 1 / (1 + distance_matrix)
            else:  # manhattan
                distance_matrix = manhattan_distances(embeddings)
                similarity_matrix = 1 / (1 + distance_matrix)
        
        titles = [doc.get('title', f'Doc {i}')[:30] for i, doc in enumerate(documents)]
        
        return {
            'matrix': similarity_matrix,
            'titles': titles,
            'metric': similarity_metric
        }
        
    except Exception as e:
        logger.error(f"Matrix generation error: {e}")
        return None

def create_relationship_heatmap(matrix_data: Dict) -> go.Figure:
    """Create relationship matrix heatmap."""
    matrix = matrix_data['matrix']
    titles = matrix_data['titles']
    metric = matrix_data['metric']
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=titles,
        y=titles,
        colorscale='Viridis',
        showscale=True,
        hovertemplate='<b>%{x}</b><br><b>%{y}</b><br>Similarity: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Document Relationship Matrix ({metric.title()} Similarity)",
        xaxis_title="Documents",
        yaxis_title="Documents",
        height=600,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )
    
    return fig

def render_network_statistics(network_data: Dict):
    """Render network analysis statistics."""
    G = network_data['graph']
    nodes = network_data['nodes']
    edges = network_data['edges']
    
    st.subheader("📊 Network Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Nodes", len(nodes))
    
    with col2:
        st.metric("Total Edges", len(edges))
    
    with col3:
        density = nx.density(G) if len(nodes) > 1 else 0
        st.metric("Network Density", f"{density:.3f}")
    
    with col4:
        avg_degree = sum(node['degree'] for node in nodes) / len(nodes) if nodes else 0
        st.metric("Avg Connections", f"{avg_degree:.1f}")
    
    # Additional statistics
    if len(nodes) > 0:
        degrees = [node['degree'] for node in nodes]
        weights = [edge['weight'] for edge in edges] if edges else []
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Connection Distribution:**")
            st.write(f"- Max connections: {max(degrees) if degrees else 0}")
            st.write(f"- Min connections: {min(degrees) if degrees else 0}")
            st.write(f"- Median connections: {np.median(degrees) if degrees else 0:.1f}")
        
        with col2:
            if weights:
                st.write("**Edge Weight Distribution:**")
                st.write(f"- Max similarity: {max(weights):.3f}")
                st.write(f"- Min similarity: {min(weights):.3f}")
                st.write(f"- Median similarity: {np.median(weights):.3f}")

def render_cluster_analysis(vector_data: Dict):
    """Render cluster analysis details."""
    clusters = vector_data['clusters']
    titles = vector_data['titles']
    
    st.subheader("🎯 Cluster Analysis")
    
    # Cluster statistics
    unique_clusters = np.unique(clusters)
    cluster_sizes = [np.sum(clusters == c) for c in unique_clusters]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Clusters", len(unique_clusters))
    
    with col2:
        st.metric("Largest Cluster", max(cluster_sizes) if cluster_sizes else 0)
    
    with col3:
        st.metric("Smallest Cluster", min(cluster_sizes) if cluster_sizes else 0)
    
    # Cluster details
    for cluster_id in unique_clusters:
        with st.expander(f"Cluster {cluster_id} ({np.sum(clusters == cluster_id)} documents)"):
            cluster_docs = [titles[i] for i, c in enumerate(clusters) if c == cluster_id]
            for doc in cluster_docs[:10]:  # Show first 10
                st.write(f"• {doc}")
            if len(cluster_docs) > 10:
                st.write(f"... and {len(cluster_docs) - 10} more documents")

def render_matrix_statistics(matrix_data: Dict):
    """Render relationship matrix statistics."""
    matrix = matrix_data['matrix']
    metric = matrix_data['metric']
    
    st.subheader("📊 Matrix Statistics")
    
    # Remove diagonal (self-similarity)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    off_diagonal = matrix[mask]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Matrix Size", f"{matrix.shape[0]}×{matrix.shape[1]}")
    
    with col2:
        st.metric("Max Similarity", f"{np.max(off_diagonal):.3f}")
    
    with col3:
        st.metric("Min Similarity", f"{np.min(off_diagonal):.3f}")
    
    with col4:
        st.metric("Avg Similarity", f"{np.mean(off_diagonal):.3f}")
    
    # Distribution histogram
    fig = px.histogram(
        x=off_diagonal,
        title=f"Similarity Distribution ({metric.title()})",
        labels={'x': 'Similarity Score', 'y': 'Frequency'},
        nbins=30
    )
    st.plotly_chart(fig, use_container_width=True)

def perform_document_clustering(data_access, collection: str, method: str, 
                              n_clusters: int, min_cluster_size: int) -> Optional[Dict]:
    """Perform document clustering analysis."""
    try:
        documents = data_access.get_documents_with_embeddings(collection, limit=500)
        if not documents or len(documents) < n_clusters:
            return None
        
        embeddings = np.array([doc.get('embedding', []) for doc in documents])
        if embeddings.size == 0:
            return None
        
        # Perform clustering based on method
        if method == "kmeans":
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings)
        elif method == "hierarchical":
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(embeddings)
        else:  # dbscan
            from sklearn.cluster import DBSCAN
            clusterer = DBSCAN(eps=0.5, min_samples=min_cluster_size)
            cluster_labels = clusterer.fit_predict(embeddings)
        
        # Filter out noise (label -1 in DBSCAN)
        valid_indices = cluster_labels != -1
        if not np.any(valid_indices):
            return None
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings[valid_indices])
        
        titles = [doc.get('title', f'Document {i}')[:50] for i, doc in enumerate(documents) if valid_indices[i]]
        contents = [doc.get('content', '')[:200] for i, doc in enumerate(documents) if valid_indices[i]]
        valid_clusters = cluster_labels[valid_indices]
        
        return {
            'embeddings': reduced_embeddings,
            'clusters': valid_clusters,
            'titles': titles,
            'contents': contents,
            'method': method,
            'pca_explained_variance': pca.explained_variance_ratio_
        }
        
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        return None

def create_cluster_visualization(cluster_data: Dict) -> go.Figure:
    """Create cluster visualization plot."""
    embeddings = cluster_data['embeddings']
    clusters = cluster_data['clusters']
    titles = cluster_data['titles']
    contents = cluster_data['contents']
    method = cluster_data['method']
    
    fig = go.Figure()
    
    # Create scatter plot with cluster colors
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        fig.add_trace(go.Scatter(
            x=embeddings[mask, 0],
            y=embeddings[mask, 1],
            mode='markers',
            name=f'Cluster {cluster_id}',
            text=[titles[i] for i in range(len(titles)) if mask[i]],
            customdata=[contents[i] for i in range(len(contents)) if mask[i]],
            hovertemplate='<b>%{text}</b><br>Cluster: %{fullData.name}<br>%{customdata}<extra></extra>',
            marker=dict(size=8, opacity=0.7)
        ))
    
    explained_var = cluster_data['pca_explained_variance']
    fig.update_layout(
        title=f"Document Clustering ({method.title()})<br><sub>PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%} variance explained</sub>",
        xaxis_title=f"Principal Component 1 ({explained_var[0]:.1%})",
        yaxis_title=f"Principal Component 2 ({explained_var[1]:.1%})",
        height=600,
        showlegend=True
    )
    
    return fig

def render_cluster_details(cluster_data: Dict):
    """Render detailed cluster information."""
    clusters = cluster_data['clusters']
    titles = cluster_data['titles']
    method = cluster_data['method']
    
    st.subheader(f"🎯 {method.title()} Clustering Details")
    
    # Cluster statistics
    unique_clusters = np.unique(clusters)
    cluster_sizes = [np.sum(clusters == c) for c in unique_clusters]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Clusters", len(unique_clusters))
    
    with col2:
        st.metric("Largest Cluster", max(cluster_sizes) if cluster_sizes else 0)
    
    with col3:
        avg_size = np.mean(cluster_sizes) if cluster_sizes else 0
        st.metric("Average Cluster Size", f"{avg_size:.1f}")
    
    # Individual cluster details
    for cluster_id in unique_clusters:
        cluster_size = np.sum(clusters == cluster_id)
        with st.expander(f"Cluster {cluster_id} ({cluster_size} documents)"):
            cluster_docs = [titles[i] for i, c in enumerate(clusters) if c == cluster_id]
            
            # Show all documents in smaller clusters, sample larger ones
            show_limit = 15
            for i, doc in enumerate(cluster_docs[:show_limit]):
                st.write(f"• {doc}")
            
            if len(cluster_docs) > show_limit:
                st.write(f"... and {len(cluster_docs) - show_limit} more documents")