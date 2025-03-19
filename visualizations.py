#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import streamlit.components.v1 as components

"""
Visualizations Module - Centralized visualization functions for ENI Magazine data analysis
This module contains all the chart generation functions used by both the main app and the simple CSV analyzer.
"""

def plot_magazine_distribution(df):
    """
    Plot distribution of articles by magazine
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    
    Returns:
    plotly.graph_objects.Figure: A plotly figure object with the magazine distribution chart
    """
    magazine_counts = df['magazine'].value_counts().reset_index()
    magazine_counts.columns = ['Magazine', 'Count']
    
    # Create a bar chart using Plotly
    fig = px.bar(
        magazine_counts,
        x='Magazine',
        y='Count',
        color='Magazine',
        title='Articles by Magazine',
        text='Count'
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Magazine',
        yaxis_title='Number of Articles',
        showlegend=False,
        hovermode='closest'
    )
    
    # Add count labels
    fig.update_traces(
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Articles: %{y}'
    )
    
    return fig

def plot_language_distribution(df):
    """
    Plot pie chart of language distribution and stacked bar chart of languages by magazine
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    
    Returns:
    tuple: Two plotly figure objects (pie chart, stacked bar chart)
    """
    if 'language' not in df.columns:
        return None, None
    
    # Prepare data for language chart
    language_counts = df['language'].value_counts().reset_index()
    language_counts.columns = ['Language', 'Count']
    
    # Create a pie chart using Plotly
    pie_fig = px.pie(
        language_counts, 
        values='Count', 
        names='Language',
        title='Articles by Language',
        color='Language',
        color_discrete_map={'ITA': '#1f77b4', 'ENG': '#ff7f0e', 'Unknown': '#cccccc'},
        hole=0.4,  # Donut chart style
    )
    
    # Customize layout
    pie_fig.update_layout(
        legend_title_text='Language',
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        annotations=[dict(text='Language<br>Distribution', x=0.5, y=0.5, font_size=15, showarrow=False)]
    )
    
    # Update traces
    pie_fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+value',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    
    # Group by magazine and language for stacked bar chart
    magazine_language = df.groupby(['magazine', 'language']).size().reset_index()
    magazine_language.columns = ['Magazine', 'Language', 'Count']
    
    # Create a stacked bar chart
    stacked_fig = px.bar(
        magazine_language,
        x='Magazine',
        y='Count',
        color='Language',
        color_discrete_map={'ITA': '#1f77b4', 'ENG': '#ff7f0e', 'Unknown': '#cccccc'},
        title='Language Distribution by Magazine',
        barmode='stack'
    )
    
    # Customize layout
    stacked_fig.update_layout(
        xaxis_title='Magazine',
        yaxis_title='Number of Articles',
        legend_title='Language',
        hovermode='closest'
    )
    
    return pie_fig, stacked_fig

def plot_theme_distribution(df, top_n=10, orientation='h'):
    """
    Plot distribution of themes as a bar chart
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    top_n (int): Number of top themes to display
    orientation (str): 'h' for horizontal or 'v' for vertical bar chart
    
    Returns:
    plotly.graph_objects.Figure: A plotly figure object with the theme distribution chart
    """
    # Prepare data for theme chart
    theme_counts = df['theme'].value_counts().head(top_n).reset_index()
    theme_counts.columns = ['Theme', 'Count']
    theme_counts = theme_counts.sort_values('Count', ascending=(orientation == 'h'))
    
    if orientation == 'h':
        # Create a horizontal bar chart using Plotly
        fig = px.bar(
            theme_counts,
            y='Theme',
            x='Count',
            color='Count',
            color_continuous_scale='Blues',
            title='Articles by Theme',
            orientation='h'
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title='Number of Articles',
            yaxis_title='Theme',
            yaxis=dict(categoryorder='total ascending'),
            coloraxis_showscale=False,
            hovermode='closest'
        )
        
        # Add count labels
        fig.update_traces(
            texttemplate='%{x}',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Articles: %{x}'
        )
    else:
        # Create a vertical bar chart using Plotly
        fig = px.bar(
            theme_counts,
            x='Theme',
            y='Count',
            color='Count',
            color_continuous_scale='Blues',
            title='Articles by Theme',
            text='Count'
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title='Theme',
            yaxis_title='Number of Articles',
            coloraxis_showscale=False,
            hovermode='closest'
        )
        
        # Add count labels
        fig.update_traces(
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Articles: %{y}'
        )
    
    return fig

def plot_theme_pie(df, title="Theme Distribution"):
    """
    Plot pie chart of theme distribution
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    title (str): Chart title
    
    Returns:
    plotly.graph_objects.Figure: A plotly figure object with the theme pie chart
    """
    theme_counts = df['theme'].value_counts()
    
    # Create a pie chart using Plotly
    fig = px.pie(
        values=theme_counts.values,
        names=theme_counts.index,
        title=title,
        hole=0.4,  # Donut style chart
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    
    # Customize layout
    fig.update_layout(
        legend_title="Themes",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    
    # Update traces
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+value'
    )
    
    return fig

def plot_top_authors(df, top_n=10):
    """
    Plot bar chart of top authors by article count
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    top_n (int): Number of top authors to display
    
    Returns:
    plotly.graph_objects.Figure: A plotly figure object with the top authors chart
    """
    # Prepare data for authors chart
    author_counts = df['author'].value_counts().head(top_n).reset_index()
    author_counts.columns = ['Author', 'Count']
    
    # Create a horizontal bar chart using Plotly
    fig = px.bar(
        author_counts,
        y='Author',
        x='Count',
        color='Count',
        color_continuous_scale='Oranges',
        title=f'Top {top_n} Authors by Article Count',
        orientation='h'
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Number of Articles',
        yaxis_title='Author',
        yaxis=dict(categoryorder='total ascending'),
        coloraxis_showscale=False,
        hovermode='closest'
    )
    
    # Add count labels
    fig.update_traces(
        texttemplate='%{x}',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Articles: %{x}'
    )
    
    return fig

def plot_issue_distribution(df):
    """
    Plot grouped bar chart of articles per issue by magazine
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    
    Returns:
    plotly.graph_objects.Figure: A plotly figure object with the issue distribution chart
    """
    # Group by magazine and issue, then count
    issue_counts = df.groupby(['magazine', 'magazine_no']).size().reset_index()
    issue_counts.columns = ['Magazine', 'Issue', 'Count']
    
    # Sort by magazine and numeric issue
    issue_counts['Issue_Numeric'] = pd.to_numeric(issue_counts['Issue'], errors='coerce')
    issue_counts = issue_counts.sort_values(['Magazine', 'Issue_Numeric'])
    
    # Create a grouped bar chart using Plotly
    fig = px.bar(
        issue_counts,
        x='Issue',
        y='Count',
        color='Magazine',
        title="Articles by Issue",
        labels={"Count": "Number of Articles"},
        barmode='group',
        hover_data=['Magazine', 'Issue', 'Count']
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Issue Number",
        yaxis_title="Number of Articles",
        hovermode="closest"
    )
    
    # Add facet for separate magazine display
    fig = px.bar(
        issue_counts, 
        x='Issue', 
        y='Count', 
        color='Magazine',
        facet_row='Magazine', 
        labels={"Count": "Number of Articles"},
        height=600,
        title="Articles per Issue by Magazine"
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Issue Number",
        yaxis_title="Number of Articles",
        legend_title="Magazine",
        hovermode="closest"
    )
    
    # Add text annotations
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    
    return fig

def plot_magazine_issues(df, magazine):
    """
    Plot bar chart of articles per issue for a specific magazine
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    magazine (str): Magazine name to filter on
    
    Returns:
    matplotlib.figure.Figure: A matplotlib figure with the issue distribution chart
    """
    # Filter for the specific magazine
    magazine_df = df[df['magazine'] == magazine]
    
    # Group by issue and count
    issue_counts = magazine_df.groupby('magazine_no').size().reset_index()
    issue_counts.columns = ['Issue', 'Count']
    
    # Sort by numeric issue
    issue_counts['Issue_Numeric'] = pd.to_numeric(issue_counts['Issue'], errors='coerce')
    issue_counts = issue_counts.sort_values('Issue_Numeric')
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(issue_counts['Issue'], issue_counts['Count'], color='#1f77b4')
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                str(int(height)), ha='center', va='bottom')
    
    ax.set_xlabel('Issue Number')
    ax.set_ylabel('Number of Articles')
    ax.set_title(f'Articles per Issue in {magazine}')
    
    # Use thinner bars with some spacing
    plt.xticks(rotation=45)
    
    return fig

def plot_format_distribution(df):
    """
    Plot distribution of article formats
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    
    Returns:
    plotly.graph_objects.Figure: A plotly figure object with the format distribution chart
    """
    if 'format' not in df.columns:
        return None
        
    format_counts = df['format'].value_counts()
    
    # Create a horizontal bar chart using Plotly
    fig = px.bar(
        y=format_counts.index,
        x=format_counts.values,
        title="Article Formats",
        labels={"x": "Number of Articles", "y": "Format"},
        color=format_counts.values,
        color_continuous_scale="Oranges",
        orientation='h'
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Number of Articles",
        yaxis_title="Format",
        coloraxis_showscale=False,
        hovermode="closest"
    )
    
    # Add count labels
    fig.update_traces(
        texttemplate="%{x}",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Articles: %{x}"
    )
    
    return fig

def plot_format_pie(df):
    """
    Plot pie chart of format distribution
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    
    Returns:
    plotly.graph_objects.Figure: A plotly figure object with the format pie chart
    """
    if 'format' not in df.columns:
        return None
        
    format_counts = df['format'].value_counts()
    
    # Create pie chart using Plotly
    fig = px.pie(
        values=format_counts.values,
        names=format_counts.index,
        title='Article Formats',
        hole=0.3
    )
    
    # Customize layout
    fig.update_layout(
        legend_title="Format",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    
    return fig

def plot_format_by_magazine(df):
    """
    Plot format distribution by magazine as stacked bar charts
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    
    Returns:
    plotly.graph_objects.Figure: A plotly figure object with the format by magazine chart
    """
    if 'format' not in df.columns:
        return None
    
    # Group by magazine and format, then count
    format_counts = df.groupby(['magazine', 'format']).size().reset_index()
    format_counts.columns = ['Magazine', 'Format', 'Count']
    
    # Create a stacked bar chart
    fig = px.bar(
        format_counts,
        x='Magazine',
        y='Count',
        color='Format',
        title='Format Distribution by Magazine',
        barmode='stack',
        labels={"Count": "Number of Articles"}
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Magazine",
        yaxis_title="Number of Articles",
        legend_title="Format",
        hovermode="closest"
    )
    
    return fig

def plot_theme_trends(df, magazine, selected_themes):
    """
    Plot line chart of theme trends across issues for a specific magazine
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    magazine (str): Magazine name to filter on
    selected_themes (list): List of themes to include
    
    Returns:
    matplotlib.figure.Figure: A matplotlib figure with the theme trends chart or None if not enough data
    """
    # Filter to selected magazine
    magazine_df = df[df['magazine'] == magazine].copy()
    
    if len(magazine_df) == 0:
        return None
    
    # Ensure theme is a string for consistency
    magazine_df['theme_str'] = magazine_df['theme'].astype(str)
    
    # Filter to selected themes
    theme_df = magazine_df[magazine_df['theme_str'].isin(selected_themes)]
    
    if len(theme_df) == 0:
        return None
    
    try:
        # Group by issue and theme
        theme_issue_counts = theme_df.groupby(['magazine_no', 'theme']).size().unstack().fillna(0)
        
        if len(theme_issue_counts) == 0:
            return None
        
        # Sort by issue number
        theme_issue_counts['issue_numeric'] = pd.to_numeric(theme_issue_counts.index, errors='coerce')
        theme_issue_counts = theme_issue_counts.sort_values('issue_numeric')
        theme_issue_counts = theme_issue_counts.drop(columns=['issue_numeric'])
        
        # Create a line chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for theme in theme_issue_counts.columns:
            ax.plot(theme_issue_counts.index, theme_issue_counts[theme], marker='o', label=theme)
        
        ax.set_xlabel('Issue Number')
        ax.set_ylabel('Number of Articles')
        ax.set_title(f'Theme Trends Across {magazine} Issues')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    except:
        return None

def plot_theme_trends_plotly(df, magazine, selected_themes):
    """
    Plot line chart of theme trends across issues using Plotly
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    magazine (str): Magazine name to filter on
    selected_themes (list): List of themes to include
    
    Returns:
    plotly.graph_objects.Figure: A plotly figure with the theme trends chart or None if not enough data
    """
    # Filter to selected magazine
    magazine_df = df[df['magazine'] == magazine].copy()
    
    if len(magazine_df) == 0:
        return None
    
    # Ensure theme is a string for consistency
    magazine_df['theme_str'] = magazine_df['theme'].astype(str)
    
    # Filter to selected themes
    theme_df = magazine_df[magazine_df['theme_str'].isin(selected_themes)]
    
    if len(theme_df) == 0:
        return None
    
    try:
        # Group by issue and theme
        issue_theme_counts = theme_df.groupby(['magazine_no', 'theme']).size().reset_index()
        issue_theme_counts.columns = ['Issue', 'Theme', 'Count']
        
        # Sort by issue number
        issue_theme_counts['Issue_Numeric'] = pd.to_numeric(issue_theme_counts['Issue'], errors='coerce')
        issue_theme_counts = issue_theme_counts.sort_values('Issue_Numeric')
        
        # Create a line chart using Plotly
        fig = px.line(
            issue_theme_counts,
            x='Issue',
            y='Count',
            color='Theme',
            markers=True,
            title=f'Theme Trends Across {magazine} Issues',
            labels={"Count": "Number of Articles"}
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title="Issue Number",
            yaxis_title="Number of Articles",
            legend_title="Theme",
            hovermode="closest"
        )
        
        # Add grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
    except:
        return None

def show_overview_charts(df):
    """
    Create a combined set of overview charts
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    
    Returns:
    dict: Dictionary of plotly figures for different chart types
    """
    charts = {
        'magazine_distribution': plot_magazine_distribution(df),
        'format_distribution': plot_format_distribution(df),
        'format_pie': plot_format_pie(df),
        'format_by_magazine': plot_format_by_magazine(df),
        'theme_distribution': plot_theme_distribution(df),
        'top_authors': plot_top_authors(df),
        'issue_distribution': plot_issue_distribution(df)
    }
    
    # Add language charts if language column exists
    if 'language' in df.columns:
        language_pie, language_stacked = plot_language_distribution(df)
        charts['language_pie'] = language_pie
        charts['language_stacked'] = language_stacked
    
    return charts 

def build_keyword_network(df):
    """
    Build a network graph data structure from article keywords
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the magazine data
    
    Returns:
    dict: Network data with nodes and links for D3 visualization
    """
    if 'keywords' not in df.columns:
        return {"nodes": [], "links": []}
    
    # Extract all keywords
    all_keywords = set()
    article_keywords = []
    themes = set(df['theme'].dropna().unique()) if 'theme' in df.columns else set()
    
    # Process each article
    for _, row in df.iterrows():
        if pd.isna(row.get('keywords')):
            continue
            
        # Split keywords and clean them
        keywords = [k.strip() for k in str(row['keywords']).split(',') if k.strip()]
        if not keywords:
            continue
            
        article_keywords.append(keywords)
        all_keywords.update(keywords)
    
    # Create nodes
    nodes = [{"id": kw, "isTheme": kw in themes} for kw in all_keywords]
    
    # Create links (edges between co-occurring keywords)
    links = []
    link_counts = {}  # To keep track of weights
    
    for keywords in article_keywords:
        # Create pairs of keywords
        for i, kw1 in enumerate(keywords):
            for kw2 in keywords[i+1:]:
                # Create a consistent key for the pair
                pair = tuple(sorted([kw1, kw2]))
                
                if pair in link_counts:
                    link_counts[pair] += 1
                else:
                    link_counts[pair] = 1
    
    # Convert the count dict to links array
    for (source, target), weight in link_counts.items():
        links.append({
            "source": source,
            "target": target,
            "weight": weight
        })
    
    return {"nodes": nodes, "links": links}

def render_keyword_network(network_data):
    """
    Render a D3 force-directed graph of keyword co-occurrences with interactive controls
    
    Parameters:
    network_data (dict): Dictionary with nodes and links data
    
    Returns:
    None: Renders directly to the Streamlit app
    """
    # Convert to JSON for embedding in HTML
    network_data_json = json.dumps(network_data)
    
    # D3 visualization HTML template with added controls
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <script src="https://d3js.org/d3.v7.min.js"></script>
      <style>
        body { 
            margin: 0; 
            padding: 0; 
            font-family: 'Helvetica Neue', Arial, sans-serif;
            color: #444;
        }
        .container {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .controls {
            padding: 12px;
            background: #f5f5f9;
            border-bottom: 1px solid #ddd;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            min-width: 120px;
        }
        .control-group label {
            font-size: 12px;
            margin-bottom: 4px;
            font-weight: bold;
            color: #555;
        }
        .search-container {
            position: relative;
            display: flex;
            flex-direction: column;
        }
        #search-input {
            padding: 6px 8px;
            width: 180px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        #search-results {
            position: absolute;
            top: 100%;
            left: 0;
            width: 180px;
            max-height: 200px;
            overflow-y: auto;
            background: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            z-index: 1000;
            display: none;
        }
        .search-result {
            padding: 4px 8px;
            cursor: pointer;
        }
        .search-result:hover {
            background: #f0f0f0;
        }
        select, input[type="range"], button {
            padding: 6px 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        button {
            background: #2e86c1;
            color: white;
            border: none;
            cursor: pointer;
            transition: background 0.2s;
            padding: 6px 12px;
        }
        button:hover {
            background: #1a5276;
        }
        svg { 
            width: 100%; 
            height: 580px; 
            background-color: #fafafa;
        }
        .viz-container {
            position: relative;
            flex-grow: 1;
        }
        .node { 
            cursor: pointer; 
        }
        .node.highlighted circle {
            stroke: #ff4c4c !important;
            stroke-width: 3px !important;
        }
        .node.dimmed {
            opacity: 0.2;
        }
        .link { 
            stroke: #999; 
            stroke-opacity: 0.6; 
        }
        .link.highlighted {
            stroke: #ff4c4c !important;
            stroke-opacity: 1 !important;
        }
        .link.dimmed {
            opacity: 0.1;
        }
        .node-label { 
            font-family: sans-serif; 
            font-size: 12px; 
            pointer-events: none;
            text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, 0 -1px 0 #fff, -1px 0 0 #fff;
        }
        .legend {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            font-size: 12px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .tooltip {
            position: absolute;
            padding: 8px 12px;
            background: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
            pointer-events: none;
            z-index: 1000;
            font-size: 12px;
            display: none;
        }
        .info-panel {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            font-size: 12px;
            max-width: 300px;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="controls">
            <div class="control-group">
                <label for="layout-select">Layout</label>
                <select id="layout-select">
                    <option value="force" selected>Force-directed</option>
                    <option value="circular">Circular</option>
                    <option value="radial">Radial</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="min-links">Min Connections</label>
                <input type="range" id="min-links" min="1" max="5" value="1" step="1">
                <span id="min-links-value">1</span>
            </div>
            
            <div class="control-group">
                <label for="node-size">Node Size</label>
                <input type="range" id="node-size" min="0.5" max="2" value="1" step="0.1">
                <span id="node-size-value">1</span>
            </div>
            
            <div class="control-group">
                <label for="color-scheme">Color Scheme</label>
                <select id="color-scheme">
                    <option value="default" selected>Default</option>
                    <option value="category10">Category 10</option>
                    <option value="accent">Accent</option>
                    <option value="paired">Paired</option>
                </select>
            </div>
            
            <div class="search-container">
                <label for="search-input">Search Keyword</label>
                <input type="text" id="search-input" placeholder="Type to search...">
                <div id="search-results"></div>
            </div>
            
            <div class="control-group">
                <label>&nbsp;</label>
                <button id="reset-button">Reset View</button>
            </div>
            
            <div class="control-group">
                <label>&nbsp;</label>
                <button id="export-button">Export SVG</button>
            </div>
        </div>
        
        <div class="viz-container">
            <svg></svg>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #1f77b4;"></div>
                    <span>Keyword</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #ff7f0e;"></div>
                    <span>Theme</span>
                </div>
                <div class="legend-item">
                    <span>Size = Connection strength</span>
                </div>
            </div>
            
            <div class="tooltip"></div>
            
            <div class="info-panel">
                <p><strong>Interactive Controls:</strong></p>
                <p>• Drag nodes to reposition</p>
                <p>• Scroll to zoom in/out</p>
                <p>• Click on nodes to highlight connections</p>
                <p>• Double-click to fix/release node position</p>
            </div>
        </div>
      </div>
      
      <script>
        const network = NETWORK_DATA_PLACEHOLDER;
        
        // Calculate node frequency and link connections
        const nodeFrequency = {};
        const nodeConnections = {};
        
        // Initialize network visualization variables
        let simulation, svg, g;
        let link, node, nodeCircles, nodeLabels;
        let width, height;
        let currentLayout = 'force';
        let colorScheme = 'default';
        let nodeScale = 1;
        let minLinks = 1;
        
        // Filter out nodes that have no connections
        const nodeIds = new Set();
        network.links.forEach(link => {
            nodeIds.add(link.source);
            nodeIds.add(link.target);
        });
        network.nodes = network.nodes.filter(node => nodeIds.has(node.id));
        
        // Calculate node frequency and connections
        network.nodes.forEach(node => {
            nodeConnections[node.id] = [];
        });
        
        network.links.forEach(link => {
            nodeFrequency[link.source] = (nodeFrequency[link.source] || 0) + link.weight;
            nodeFrequency[link.target] = (nodeFrequency[link.target] || 0) + link.weight;
            
            // Store connections for highlight
            if (nodeConnections[link.source]) {
                nodeConnections[link.source].push(link.target);
            }
            if (nodeConnections[link.target]) {
                nodeConnections[link.target].push(link.source);
            }
        });
        
        // Setup the visualization
        function initializeVisualization() {
            // Set dimensions based on container
            const container = document.querySelector('.viz-container');
            width = container.clientWidth;
            height = container.clientHeight;
            
            // Create SVG
            svg = d3.select("svg")
                .attr("viewBox", [0, 0, width, height]);
            
            // Create a container group for zoom/pan
            g = svg.append("g");
            
            // Add zoom behavior
            svg.call(d3.zoom()
                .extent([[0, 0], [width, height]])
                .scaleExtent([0.1, 8])
                .on("zoom", (event) => {
                    g.attr("transform", event.transform);
                }));
            
            // Create simulation
            simulation = d3.forceSimulation(network.nodes)
                .force("link", d3.forceLink(network.links)
                    .id(d => d.id)
                    .distance(70))
                .force("charge", d3.forceManyBody()
                    .strength(-100))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide()
                    .radius(d => getNodeRadius(d) + 2));
            
            // Draw links
            link = g.append("g")
                .selectAll("line")
                .data(network.links)
                .join("line")
                .attr("class", "link")
                .attr("stroke-width", d => Math.sqrt(d.weight));
            
            // Create node group
            node = g.append("g")
                .selectAll("g")
                .data(network.nodes)
                .join("g")
                .attr("class", "node")
                .call(drag(simulation))
                .on("click", highlightConnections)
                .on("dblclick", toggleFixNode)
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip);
            
            // Add circles to nodes
            nodeCircles = node.append("circle")
                .attr("r", getNodeRadius)
                .attr("fill", getNodeColor)
                .attr("stroke", "#fff")
                .attr("stroke-width", 1.5);
            
            // Add text labels to nodes
            nodeLabels = node.append("text")
                .attr("class", "node-label")
                .attr("dx", d => getNodeRadius(d) + 5)
                .attr("dy", ".35em")
                .text(d => d.id);
            
            // Add tooltips
            node.append("title")
                .text(d => `${d.id}${d.isTheme ? " (Theme)" : ""}`);
            
            // Update positions on each tick
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node.attr("transform", function(d) {
                    return "translate(" + d.x + "," + d.y + ")";
                });
            });
            
            setupControls();
            setupSearch();
        }
        
        // Utility functions
        function getNodeRadius(d) {
            return (Math.sqrt(nodeFrequency[d.id] || 1) * 3 + 5) * nodeScale;
        }
        
        function getNodeColor(d) {
            if (colorScheme === 'default') {
                return d.isTheme ? "#ff7f0e" : "#1f77b4";
            } else if (colorScheme === 'category10') {
                const color = d3.scaleOrdinal(d3.schemeCategory10);
                return d.isTheme ? color(d.id) : color("keyword_" + d.id);
            } else if (colorScheme === 'accent') {
                const color = d3.scaleOrdinal(d3.schemeAccent);
                return d.isTheme ? color(d.id) : color("keyword_" + d.id);
            } else if (colorScheme === 'paired') {
                const color = d3.scaleOrdinal(d3.schemePaired);
                return d.isTheme ? color(d.id) : color("keyword_" + d.id);
            }
        }
        
        function setupControls() {
            // Layout selector
            d3.select("#layout-select").on("change", function() {
                currentLayout = this.value;
                updateLayout();
            });
            
            // Min links filter
            d3.select("#min-links").on("input", function() {
                minLinks = +this.value;
                d3.select("#min-links-value").text(minLinks);
                filterByLinks();
            });
            
            // Node size selector
            d3.select("#node-size").on("input", function() {
                nodeScale = +this.value;
                d3.select("#node-size-value").text(parseFloat(nodeScale).toFixed(1));
                updateNodeSizes();
            });
            
            // Color scheme selector
            d3.select("#color-scheme").on("change", function() {
                colorScheme = this.value;
                updateColors();
            });
            
            // Reset button
            d3.select("#reset-button").on("click", resetVisualization);
            
            // Export button
            d3.select("#export-button").on("click", exportSVG);
        }
        
        function setupSearch() {
            const searchInput = d3.select("#search-input");
            const searchResults = d3.select("#search-results");
            
            searchInput.on("input", function() {
                const query = this.value.toLowerCase();
                if (query.length < 2) {
                    searchResults.style("display", "none");
                    resetHighlights();
                    return;
                }
                
                const matches = network.nodes
                    .filter(d => d.id.toLowerCase().includes(query))
                    .slice(0, 10);
                
                if (matches.length === 0) {
                    searchResults.style("display", "none");
                    return;
                }
                
                searchResults.style("display", "block")
                    .html("")
                    .selectAll("div")
                    .data(matches)
                    .enter()
                    .append("div")
                    .attr("class", "search-result")
                    .text(d => d.id)
                    .on("click", function(event, d) {
                        searchInput.property("value", d.id);
                        searchResults.style("display", "none");
                        highlightNode(d);
                        centerNode(d);
                    });
            });
            
            // Close search results when clicking outside
            document.addEventListener("click", function(event) {
                if (!event.target.closest(".search-container")) {
                    searchResults.style("display", "none");
                }
            });
        }
        
        function filterByLinks() {
            // Filter nodes by number of connections
            const filteredNodes = network.nodes.filter(d => {
                const connectionCount = nodeConnections[d.id]?.length || 0;
                return connectionCount >= minLinks;
            });
            
            // Filter links to only include filtered nodes
            const filteredNodeIds = new Set(filteredNodes.map(d => d.id));
            const filteredLinks = network.links.filter(d => 
                filteredNodeIds.has(d.source.id || d.source) && 
                filteredNodeIds.has(d.target.id || d.target)
            );
            
            // Update the simulation with filtered data
            simulation.nodes(filteredNodes);
            simulation.force("link").links(filteredLinks);
            
            // Update the visualization
            link = link.data(filteredLinks, d => `${d.source.id || d.source}-${d.target.id || d.target}`);
            link.exit().remove();
            link = link.enter().append("line")
                .attr("class", "link")
                .attr("stroke-width", d => Math.sqrt(d.weight))
                .merge(link);
            
            node = node.data(filteredNodes, d => d.id);
            node.exit().remove();
            const newNodes = node.enter().append("g")
                .attr("class", "node")
                .call(drag(simulation))
                .on("click", highlightConnections)
                .on("dblclick", toggleFixNode)
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip);
                
            newNodes.append("circle")
                .attr("r", getNodeRadius)
                .attr("fill", getNodeColor)
                .attr("stroke", "#fff")
                .attr("stroke-width", 1.5);
                
            newNodes.append("text")
                .attr("class", "node-label")
                .attr("dx", d => getNodeRadius(d) + 5)
                .attr("dy", ".35em")
                .text(d => d.id);
                
            newNodes.append("title")
                .text(d => `${d.id}${d.isTheme ? " (Theme)" : ""}`);
                
            node = newNodes.merge(node);
            nodeCircles = node.select("circle");
            nodeLabels = node.select("text");
            
            // Restart the simulation
            simulation.alpha(1).restart();
        }
        
        function updateLayout() {
            if (currentLayout === 'force') {
                simulation
                    .force("link", d3.forceLink(network.links)
                        .id(d => d.id)
                        .distance(70))
                    .force("charge", d3.forceManyBody()
                        .strength(-100))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("collision", d3.forceCollide()
                        .radius(d => getNodeRadius(d) + 2));
            } else if (currentLayout === 'circular') {
                // Arrange nodes in a circle
                simulation
                    .force("link", d3.forceLink(network.links)
                        .id(d => d.id)
                        .distance(50))
                    .force("charge", d3.forceManyBody()
                        .strength(-50))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("x", d3.forceX(width / 2).strength(0.1))
                    .force("y", d3.forceY(height / 2).strength(0.1))
                    .force("r", d3.forceRadial(Math.min(width, height) * 0.4, width / 2, height / 2).strength(1));
            } else if (currentLayout === 'radial') {
                // Cluster nodes by type (theme vs keyword)
                const types = Array.from(new Set(network.nodes.map(d => d.isTheme ? "theme" : "keyword")));
                const typeScale = d3.scalePoint()
                    .domain(types)
                    .range([1, 3])
                    .padding(1);
                
                simulation
                    .force("link", d3.forceLink(network.links)
                        .id(d => d.id)
                        .distance(70))
                    .force("charge", d3.forceManyBody()
                        .strength(-60))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("r", d3.forceRadial(d => typeScale(d.isTheme ? "theme" : "keyword") * 100, width / 2, height / 2).strength(0.8));
            }
            
            // Reset all fixed positions
            network.nodes.forEach(d => {
                d.fx = null;
                d.fy = null;
            });
            
            // Restart the simulation
            simulation.alpha(1).restart();
        }
        
        function updateNodeSizes() {
            nodeCircles.attr("r", getNodeRadius);
            nodeLabels.attr("dx", d => getNodeRadius(d) + 5);
            
            // Update collision detection
            simulation.force("collision", d3.forceCollide().radius(d => getNodeRadius(d) + 2));
            simulation.alpha(0.3).restart();
        }
        
        function updateColors() {
            nodeCircles.attr("fill", getNodeColor);
            
            // Also update legend colors
            const legendItems = d3.selectAll(".legend-item");
            legendItems.select(".legend-color").each(function(d, i) {
                if (i === 0) {
                    // Regular keyword
                    d3.select(this).style("background-color", getNodeColor({isTheme: false}));
                } else if (i === 1) {
                    // Theme
                    d3.select(this).style("background-color", getNodeColor({isTheme: true}));
                }
            });
        }
        
        function showTooltip(event, d) {
            const tooltip = d3.select(".tooltip");
            const connections = nodeConnections[d.id]?.length || 0;
            
            tooltip.html(`
                <div><strong>${d.id}</strong></div>
                <div>${d.isTheme ? "Theme" : "Keyword"}</div>
                <div>Connections: ${connections}</div>
                <div>Weight: ${nodeFrequency[d.id] || 0}</div>
            `)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 10) + "px")
            .style("display", "block");
        }
        
        function hideTooltip() {
            d3.select(".tooltip").style("display", "none");
        }
        
        function highlightConnections(event, d) {
            // Reset previous highlights
            resetHighlights();
            
            // Get connected nodes
            const connectedIds = new Set(nodeConnections[d.id] || []);
            connectedIds.add(d.id); // Add selected node too
            
            // Highlight nodes and links
            node.classed("highlighted", n => n.id === d.id);
            node.classed("dimmed", n => !connectedIds.has(n.id));
            
            link.classed("highlighted", l => 
                (l.source.id === d.id || l.target.id === d.id)
            );
            link.classed("dimmed", l => 
                !(l.source.id === d.id || l.target.id === d.id)
            );
        }
        
        function highlightNode(d) {
            // Reset previous highlights
            resetHighlights();
            
            // Highlight just this node
            node.classed("highlighted", n => n.id === d.id);
            node.classed("dimmed", n => n.id !== d.id);
            
            // Highlight links connected to this node
            link.classed("highlighted", l => 
                (l.source.id === d.id || l.target.id === d.id)
            );
            link.classed("dimmed", l => 
                !(l.source.id === d.id || l.target.id === d.id)
            );
        }
        
        function resetHighlights() {
            node.classed("highlighted", false);
            node.classed("dimmed", false);
            link.classed("highlighted", false);
            link.classed("dimmed", false);
        }
        
        function centerNode(d) {
            // Pan to center on this node
            if (d.x && d.y) {
                const transform = d3.zoomIdentity
                    .translate(width/2 - d.x, height/2 - d.y)
                    .scale(1);
                
                svg.transition()
                    .duration(750)
                    .call(
                        d3.zoom().transform,
                        transform
                    );
            }
        }
        
        function toggleFixNode(event, d) {
            if (d.fx != null) {
                // If position is fixed, release it
                d.fx = null;
                d.fy = null;
                d3.select(this).select("circle")
                    .style("stroke", "#fff");
            } else {
                // Fix the position
                d.fx = d.x;
                d.fy = d.y;
                d3.select(this).select("circle")
                    .style("stroke", "#00C853");
            }
        }
        
        function resetVisualization() {
            resetHighlights();
            
            // Reset all fixed positions
            network.nodes.forEach(d => {
                d.fx = null;
                d.fy = null;
            });
            
            // Reset controls
            d3.select("#layout-select").property("value", "force");
            currentLayout = "force";
            
            d3.select("#min-links").property("value", 1);
            d3.select("#min-links-value").text("1");
            minLinks = 1;
            
            d3.select("#node-size").property("value", 1);
            d3.select("#node-size-value").text("1");
            nodeScale = 1;
            
            d3.select("#color-scheme").property("value", "default");
            colorScheme = "default";
            
            // Reset the search
            d3.select("#search-input").property("value", "");
            d3.select("#search-results").style("display", "none");
            
            // Reset zoom
            svg.transition()
                .duration(750)
                .call(
                    d3.zoom().transform,
                    d3.zoomIdentity.translate(width/2, height/2).scale(1)
                );
            
            // Update the visualization
            updateNodeSizes();
            updateColors();
            filterByLinks();
            updateLayout();
        }
        
        function exportSVG() {
            // Create a copy of the svg with current zoom/pan state
            const svgCopy = document.querySelector("svg").cloneNode(true);
            svgCopy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
            
            // Add CSS styles for the nodes and links
            const style = document.createElement("style");
            style.textContent = `
                .node circle { fill: ${getNodeColor}; stroke: #fff; stroke-width: 1.5; }
                .link { stroke: #999; stroke-opacity: 0.6; }
                .node-label { 
                    font-family: sans-serif; 
                    font-size: 12px; 
                    text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, 0 -1px 0 #fff, -1px 0 0 #fff;
                }
            `;
            svgCopy.insertBefore(style, svgCopy.firstChild);
            
            // Create a blob with the SVG content
            const svgData = new XMLSerializer().serializeToString(svgCopy);
            const blob = new Blob([svgData], {type: "image/svg+xml"});
            
            // Create a download link
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "keyword_network.svg";
            document.body.appendChild(a);
            a.click();
            
            // Cleanup
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 0);
        }
        
        // Drag function for nodes
        function drag(simulation) {
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                // Only keep the position fixed if double-clicked
                if (!d.fixed) {
                    d.fx = null;
                    d.fy = null;
                }
            }
            
            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }
        
        // Initialize the visualization when the document is ready
        document.addEventListener("DOMContentLoaded", initializeVisualization);
        
        // Also reinitialize on window resize
        window.addEventListener("resize", () => {
            // Only completely reinitialize if the size change is significant
            const newWidth = document.querySelector('.viz-container').clientWidth;
            const newHeight = document.querySelector('.viz-container').clientHeight;
            
            if (Math.abs(newWidth - width) > 50 || Math.abs(newHeight - height) > 50) {
                // Remove existing elements
                svg.selectAll("*").remove();
                
                // Reinitialize
                initializeVisualization();
            }
        });
        
        // Initialize immediately
        initializeVisualization();
      </script>
    </body>
    </html>
    """
    
    # Replace the placeholder with the actual data
    html_string = html_template.replace('NETWORK_DATA_PLACEHOLDER', network_data_json)
    
    # Render the HTML in Streamlit
    components.html(html_string, height=750) 