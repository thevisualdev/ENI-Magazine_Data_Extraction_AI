#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple, Any, Set
# Import visualization functions from the new module
from visualizations import (
    plot_magazine_distribution,
    plot_language_distribution,
    plot_theme_distribution,
    plot_theme_pie,
    plot_top_authors,
    plot_issue_distribution,
    plot_format_distribution,
    plot_format_by_magazine,
    plot_format_pie,
    plot_theme_trends_plotly,
    show_overview_charts,
    build_keyword_network,
    render_keyword_network
)

# Set page configuration
st.set_page_config(
    page_title="ENI Magazine Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# Title and intro
st.title("ENI Magazine Data Analyzer")
st.write("""
This dashboard helps analyze the extracted magazine data from ENI Magazine.
""")

# Sidebar info
st.sidebar.title("Navigation")
st.sidebar.info(
    "Select different options to view various analyses."
)

# Load data
@st.cache_data
def load_data():
    """Load data from CSV and perform basic preprocessing"""
    csv_path = "output/extracted_data.csv"
    
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found: {csv_path}")
        st.info("Please process some articles first to generate the CSV output.")
        return None
        
    df = pd.read_csv(csv_path)
    
    # Normalize magazine_no to numeric when possible
    df['magazine_no_numeric'] = pd.to_numeric(df['magazine_no'], errors='coerce')
    
    return df

# Load the data
df = load_data()

if df is None:
    st.stop()

# Update navigation to match app.py structure
pages = [
    "📈 Overview", 
    "🔍 Data Validation",
    "📊 Theme Analysis",
    "🔗 Keywords"
]

page = st.sidebar.radio("Select Page", pages)

# Basic stats
if page == "📈 Overview":
    st.header("Data Overview")
    
    # Display basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Articles", len(df))
    with col2:
        st.metric("Magazines", df['magazine'].nunique())
    with col3:
        st.metric("Issues", df.groupby(['magazine', 'magazine_no']).ngroups)
        
    # Second row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Authors", df['author'].nunique())
    with col2:
        st.metric("Themes", df['theme'].nunique())
    with col3:
        st.metric("Formats", df['format'].nunique())
    
    # Articles per magazine
    st.subheader("Articles per Magazine")
    fig = plot_magazine_distribution(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Articles per issue - using the new faceted chart
    st.subheader("Articles per Issue")
    fig = plot_issue_distribution(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top authors
    st.subheader("Top 10 Authors")
    fig = plot_top_authors(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Format distribution chart - now with by-magazine breakdown
    st.subheader("Format Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_format_distribution(df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Format information not available.")
    
    with col2:
        fig = plot_format_by_magazine(df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Format information not available.")
    
    # Language distribution if available
    if 'language' in df.columns:
        st.subheader("Language Distribution")
        language_pie, language_stacked = plot_language_distribution(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(language_pie, use_container_width=True)
        with col2:
            st.plotly_chart(language_stacked, use_container_width=True)

elif page == "🔍 Data Validation":
    st.header("Data Validation")
    
    st.write("""
    Check data quality and validate metadata.
    """)
    
    # Define expected ranges and values
    expected_ranges = {
        'WE': list(range(34, 64)),  # 34 to 63 (expanded to include earlier issues)
        'Orizzonti': list(range(55, 65))  # 55 to 64
    }
    
    expected_counts = {
        'WE': 16,
        'Orizzonti': 12
    }
    
    # Run validation checks
    with st.spinner("Running validations..."):
        # Validate magazine names
        st.subheader("Magazine Names")
        valid_magazines = {'Orizzonti', 'WE'}
        invalid_magazines = df[~df['magazine'].isin(valid_magazines)]['magazine'].unique().tolist()
        
        if invalid_magazines:
            st.error(f"Found {len(invalid_magazines)} invalid magazine names:")
            for mag in invalid_magazines:
                st.write(f"  - '{mag}'")
        else:
            st.success("✅ All magazine names are valid (Orizzonti or WE)")
        
        # Validate issue numbers
        st.subheader("Issue Numbers")
        
        # Normalize magazine_no to integers
        df['magazine_no_norm'] = df['magazine_no'].apply(
            lambda x: int(float(x)) if pd.notnull(x) and str(x).replace('.', '', 1).isdigit() else None
        )
        
        # Check for out of range issues
        out_of_range = []
        for magazine, expected_issues in expected_ranges.items():
            magazine_df = df[df['magazine'] == magazine]
            
            for _, row in magazine_df.iterrows():
                issue = row['magazine_no_norm']
                # Skip 'Not Specified' or None values when checking ranges
                if issue is not None and issue not in expected_issues and row['magazine_no'] != 'Not Specified':
                    out_of_range.append({
                        'magazine': magazine,
                        'issue': row['magazine_no'],
                        'normalized_issue': issue
                    })
        
        if out_of_range:
            st.error(f"Found {len(out_of_range)} issues outside expected ranges:")
            for item in out_of_range:
                st.write(f"  - {item['magazine']} issue {item['issue']} (normalized: {item['normalized_issue']})")
        else:
            st.success("✅ All magazine issues are within expected ranges")
        
        # Check for missing issues
        missing_issues = {}
        for magazine, expected_issues in expected_ranges.items():
            magazine_df = df[df['magazine'] == magazine]
            
            if len(magazine_df) == 0:
                missing_issues[magazine] = expected_issues
                continue
                
            found_issues = set(magazine_df['magazine_no_norm'].dropna().unique())
            magazine_missing = [i for i in expected_issues if i not in found_issues]
            
            if magazine_missing:
                missing_issues[magazine] = magazine_missing
        
        missing_count = sum(len(issues) for issues in missing_issues.values())
        if missing_count > 0:
            st.warning(f"Missing {missing_count} expected issues:")
            for magazine, issues in missing_issues.items():
                if issues:
                    st.write(f"  - {magazine}: missing issues {', '.join(map(str, issues))}")
        else:
            st.success("✅ No missing issues detected")
        
        # Check article counts
        st.subheader("Article Counts")
        
        # Count articles per issue
        issue_counts = df.groupby(['magazine', 'magazine_no']).size()
        
        # Find issues with low article counts
        issues_with_low_counts = []
        threshold = 0.5  # 50% of expected
        
        for (magazine, issue), count in issue_counts.items():
            if magazine in expected_counts:
                expected = expected_counts[magazine]
                if count < expected * threshold:
                    issues_with_low_counts.append({
                        'magazine': magazine,
                        'issue': issue,
                        'count': count,
                        'expected': expected,
                        'percentage': round(count / expected * 100, 1)
                    })
        
        if issues_with_low_counts:
            st.warning(f"Found {len(issues_with_low_counts)} issues with fewer than 50% of expected articles:")
            for item in issues_with_low_counts:
                st.write(f"  - {item['magazine']} {item['issue']}: {item['count']} articles (only {item['percentage']}% of expected {item['expected']})")
        else:
            st.success("✅ All issues have reasonable article counts")
        
        # Check for required fields
        st.subheader("Required Fields")
        
        required_fields = [
            'author', 'title', 'magazine', 'magazine_no', 
            'abstract', 'theme', 'format', 'geographic_area', 'keywords'
        ]
        
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            st.error(f"Missing required fields: {', '.join(missing_fields)}")
        else:
            st.success("✅ All required fields exist in the dataset")
        
        # Check for null values
        null_counts = {}
        for field in required_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    null_counts[field] = null_count
        
        if null_counts:
            st.warning("Found null values in these fields:")
            for field, count in null_counts.items():
                percentage = round(count / len(df) * 100, 1)
                st.write(f"  - {field}: {count} nulls ({percentage}%)")
        else:
            st.success("✅ No null values found in required fields")

elif page == "📊 Theme Analysis":
    st.header("Theme Analysis")
    
    st.write("""
    Analyze themes and their distribution across magazines and issues.
    """)
    
    # Overall theme distribution
    st.subheader("Overall Theme Distribution")
    fig = plot_theme_distribution(df, top_n=15, orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    
    # Themes by magazine
    st.subheader("Theme Distribution by Magazine")
    
    # Create a radio button to select magazine
    magazine = st.radio("Select Magazine", sorted(df['magazine'].unique()))
    
    # Filter to selected magazine
    magazine_df = df[df['magazine'] == magazine]
    
    if len(magazine_df) == 0:
        st.warning(f"No articles found for magazine: {magazine}")
        st.stop()
    
    # Plot theme pie chart for this magazine
    fig = plot_theme_pie(magazine_df, title=f"Theme Distribution in {magazine}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Theme trends across issues - now using Plotly
    st.subheader("Theme Trends Across Issues")
    
    # Get top 5 themes for this magazine
    top_themes = magazine_df['theme'].value_counts().head(5).index.tolist()
    
    # Let user select themes to display
    selected_themes = st.multiselect(
        "Select Themes to Display",
        sorted(magazine_df['theme'].unique()),
        default=top_themes[:3] if top_themes else None
    )
    
    if selected_themes:
        # Use the new Plotly-based theme trends function
        fig = plot_theme_trends_plotly(df, magazine, selected_themes)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data to create theme trends chart for the selected themes.")
    else:
        st.info("Please select at least one theme to display.")

elif page == "🔗 Keywords":
    st.header("Keyword Network Analysis")
    
    st.write("""
    Explore relationships between keywords that appear together in articles.
    This visualization shows how keywords are connected, with the size of nodes 
    indicating how frequently a keyword is used.
    """)
    
    # Add filtering options
    min_weight = st.slider(
        "Minimum co-occurrence (link weight)", 
        min_value=1, 
        max_value=5, 
        value=1,
        help="Filter links by minimum number of times keywords appear together"
    )
    
    max_nodes = st.slider(
        "Maximum number of nodes", 
        min_value=20, 
        max_value=200, 
        value=100,
        help="Limit the number of nodes to improve performance"
    )
    
    # Build network data
    with st.spinner("Building keyword network..."):
        network_data = build_keyword_network(df)
        
        # Filter links by weight
        if min_weight > 1:
            network_data["links"] = [link for link in network_data["links"] if link["weight"] >= min_weight]
        
        # Count node connections after filtering links
        node_connections = {}
        for link in network_data["links"]:
            node_connections[link["source"]] = node_connections.get(link["source"], 0) + 1
            node_connections[link["target"]] = node_connections.get(link["target"], 0) + 1
        
        # Filter nodes by connection count and limit to max_nodes
        connected_nodes = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)
        if len(connected_nodes) > max_nodes:
            connected_nodes = connected_nodes[:max_nodes]
            
        # Keep only the top nodes
        top_node_ids = {node[0] for node in connected_nodes}
        network_data["nodes"] = [node for node in network_data["nodes"] if node["id"] in top_node_ids]
        network_data["links"] = [
            link for link in network_data["links"] 
            if link["source"] in top_node_ids and link["target"] in top_node_ids
        ]
    
    st.caption(f"Showing {len(network_data['nodes'])} keywords with {len(network_data['links'])} connections")
    
    # Render the network visualization
    with st.spinner("Rendering network visualization..."):
        render_keyword_network(network_data)
    
    # Add instructions
    with st.expander("How to use the visualization"):
        st.markdown("""
        - **Zoom**: Use mouse wheel to zoom in and out
        - **Pan**: Click and drag the background to move the view
        - **Move nodes**: Click and drag nodes to reposition them
        - **View details**: Hover over a node to see the keyword name
        - **Colors**: Orange nodes are keywords that also appear as themes
        """)
        
    # Add explanation of visualization
    with st.expander("About this visualization"):
        st.markdown("""
        This network graph represents the relationships between keywords in the magazine articles:
        
        - Each **node** represents a keyword from the articles
        - **Links** connect keywords that appear together in the same article
        - **Node size** indicates how frequently the keyword appears across articles
        - **Link thickness** shows how often the keywords appear together
        - **Colors** distinguish between regular keywords (blue) and those that also appear as themes (orange)
        
        The visualization uses a force-directed layout algorithm that places connected nodes closer together,
        revealing clusters of related keywords.
        """) 