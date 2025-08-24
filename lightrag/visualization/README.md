# LightRAG Visualization Tool

A Streamlit-based web application for exploring and analyzing LightRAG knowledge base data stored in ChromaDB.

## Features

- **Collection Overview**: View all ChromaDB collections with statistics and metadata
- **Document Browser**: Browse documents within collections with pagination
- **Search Functionality**: Full-text search across collections with relevance scoring
- **Data Export**: Export collection data in JSON and CSV formats
- **Analytics Dashboard**: Basic analytics and visualizations of knowledge patterns

## Quick Start

### Prerequisites

- Python 3.8+
- LightRAG framework with ChromaDB data
- Required dependencies (see requirements.txt)

### Installation

1. Navigate to the visualization directory:
```bash
cd lightrag/visualization/
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure LightRAG ChromaDB data is available (the tool will automatically detect it)

### Running the Application

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

## Configuration

The application can be configured via `config.yaml`:

```yaml
# Database Configuration
database:
  chromadb_path: "../knowledge/chromadb"  # Path to ChromaDB directory

# Performance Settings  
performance:
  cache_timeout: 3600      # Cache timeout in seconds
  pagination_size: 50      # Documents per page
  search_limit: 100        # Maximum search results

# Visualization Settings
visualization:
  max_nodes: 1000          # Maximum nodes in network graphs
  default_k_neighbors: 5   # Default number of neighbors for similarity
```

## Usage

### Overview Page
- View total collections and documents
- See collection summary with document counts
- Monitor database connection status

### Collections Page
- Select and explore individual collections
- View metadata fields and sample documents
- Analyze collection structure and content

### Search Page
- Search across all collections or filter by specific collections
- View search results with relevance scores
- Examine document content and metadata

### Document Browser
- Browse documents with pagination
- Navigate through large collections efficiently
- View full document content and metadata

### Analytics Page
- View document distribution across collections
- Analyze knowledge base growth patterns (Phase 3)
- Explore document relationships (Phase 3)

## Architecture

```
lightrag/visualization/
├── app.py              # Main Streamlit application
├── data_access.py      # ChromaDB interface layer
├── utils.py           # Utility functions
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── .streamlit/        # Streamlit configuration
└── README.md         # This file
```

### Key Components

- **LightRAGDataAccess**: Main interface to ChromaDB collections
- **Streamlit Pages**: Modular page rendering for different views
- **Configuration Management**: YAML-based configuration with defaults
- **Error Handling**: Comprehensive error handling and logging

## Data Sources

The tool automatically connects to:

1. **Primary**: `../knowledge/chromadb/` (RIF knowledge directory)
2. **Fallback**: `../collections/` (LightRAG collections directory)
3. **Custom**: Configurable via `config.yaml`

## Collections Supported

- **patterns**: Successful code patterns and templates
- **decisions**: Architectural decisions and rationale
- **code_snippets**: Reusable code examples and functions
- **issue_resolutions**: Resolved issues and their solutions

## Performance Considerations

- **Pagination**: Large collections are paginated for performance
- **Caching**: Frequently accessed data is cached
- **Limits**: Search results and export sizes are limited by configuration
- **Lazy Loading**: Data is loaded on-demand to reduce memory usage

## Troubleshooting

### Common Issues

1. **"Failed to initialize data access"**
   - Check that ChromaDB directory exists
   - Verify LightRAG installation
   - Check file permissions

2. **"No collections found"**
   - Ensure LightRAG has been used to store data
   - Verify ChromaDB path in configuration
   - Check that collections contain data

3. **Search not working**
   - Verify collections have embedding data
   - Check ChromaDB connection
   - Try simpler search queries

### Debugging

Enable debug logging by setting environment variable:
```bash
export LIGHTRAG_LOG_LEVEL=DEBUG
streamlit run app.py
```

View logs in the terminal or check `visualization.log` file.

## Development

### Phase 1 (Current)
- ✅ Basic Streamlit application structure
- ✅ ChromaDB data access layer
- ✅ Collection overview and browsing
- ✅ Document search functionality

### Phase 2 (Planned)
- Enhanced dashboard with statistics
- Advanced filtering and metadata search
- Data export functionality
- Performance optimizations

### Phase 3 (Planned)
- Document relationship visualization
- Vector space visualization
- Knowledge timeline view
- Interactive network graphs

### Phase 4 (Planned)
- RIF agent integration
- Real-time data updates
- GitHub issue linking
- Deployment configuration

## API Documentation

### LightRAGDataAccess Methods

```python
# Get collection overview
overview = data_access.get_collection_overview()

# Get collection details
details = data_access.get_collection_details("patterns")

# Search documents
results = data_access.search_documents("query", collection_name="patterns")

# Get paginated documents
page_data = data_access.get_documents_paginated("patterns", page=0, page_size=50)

# Export collection
export_data = data_access.export_collection("patterns", format="json")
```

## Contributing

1. Follow the existing code structure and patterns
2. Add docstrings to all functions
3. Include error handling for all operations
4. Test with different data sizes and configurations
5. Update documentation for new features

## License

This tool is part of the RIF (Reactive Intelligence Framework) project.