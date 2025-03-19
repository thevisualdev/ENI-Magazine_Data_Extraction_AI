#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple, Any, Set
import plotly.express as px
import plotly.graph_objects as go

"""
CSV Analyzer - Streamlit dashboard for analyzing and exploring the extracted CSV data
"""

def setup_page():
    # Set page configuration
    st.set_page_config(
        page_title="ENI Magazine Data Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar title and info
    st.sidebar.title("ENI Magazine Data Analyzer")
    st.sidebar.info(
        "This dashboard helps analyze and explore the extracted magazine data. "
        "Select different options to view statistics, validate data quality, "
        "and explore articles."
    )

# Load data function
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV and perform basic preprocessing"""
    df = pd.read_csv(file_path)
    
    # Normalize magazine_no to numeric when possible
    df['magazine_no_numeric'] = pd.to_numeric(df['magazine_no'], errors='coerce')
    
    return df

# Main layout
def main():
    # Check if output directory and CSV file exist
    output_dir = "output"
    csv_path = os.path.join(output_dir, "extracted_data.csv")
    
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found: {csv_path}")
        st.info("Please process some articles first to generate the CSV output.")
        return
    
    # Load the data
    df = load_data(csv_path)
    
    # Simplified sidebar navigation
    nav_options = [
        "📈 Overview",
        "🔍 Data Validation", 
        "🔎 Explorer"
    ]
    
    nav_selection = st.sidebar.radio("Navigation", nav_options)
    
    # Display content based on selection
    if nav_selection == "📈 Overview":
        show_overview(df)
    elif nav_selection == "🔍 Data Validation":
        show_validation(df)
    elif nav_selection == "🔎 Explorer":
        show_explorer(df)

def show_overview(df: pd.DataFrame):
    """Display overview statistics and charts"""
    st.title("📈 Data Overview")
    
    # Basic stats in a nice layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Articles", len(df))
    with col2:
        st.metric("Magazines", df['magazine'].nunique())
    with col3:
        st.metric("Issues", df.groupby(['magazine', 'magazine_no']).ngroups)
    
    # Second row of stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Authors", df['author'].nunique())
    with col2:
        st.metric("Themes", df['theme'].nunique())
    with col3:
        if 'language' in df.columns:
            language_counts = df['language'].value_counts()
            ita_count = language_counts.get('ITA', 0)
            eng_count = language_counts.get('ENG', 0)
            st.metric("Languages", f"ITA: {ita_count}, ENG: {eng_count}")
        else:
            st.metric("Formats", df['format'].nunique())
    
    # Articles per magazine chart
    st.subheader("Articles per Magazine")
    
    # Prepare data for chart
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
    
    # Display the interactive chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Language distribution chart (if language field exists)
    if 'language' in df.columns:
        st.subheader("Language Distribution")
        
        # Prepare data for language chart
        language_counts = df['language'].value_counts().reset_index()
        language_counts.columns = ['Language', 'Count']
        
        # Create a pie chart using Plotly
        fig = px.pie(
            language_counts, 
            values='Count', 
            names='Language',
            title='Articles by Language',
            color='Language',
            color_discrete_map={'ITA': '#1f77b4', 'ENG': '#ff7f0e', 'Unknown': '#cccccc'},
            hole=0.4,  # Donut chart style
        )
        
        # Customize layout
        fig.update_layout(
            legend_title_text='Language',
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            annotations=[dict(text='Language<br>Distribution', x=0.5, y=0.5, font_size=15, showarrow=False)]
        )
        
        # Update traces
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hoverinfo='label+percent+value',
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        
        # Display the interactive chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a stacked bar chart showing languages by magazine
        st.subheader("Languages by Magazine")
        
        # Group by magazine and language
        magazine_language = df.groupby(['magazine', 'language']).size().reset_index()
        magazine_language.columns = ['Magazine', 'Language', 'Count']
        
        # Create a stacked bar chart
        fig = px.bar(
            magazine_language,
            x='Magazine',
            y='Count',
            color='Language',
            color_discrete_map={'ITA': '#1f77b4', 'ENG': '#ff7f0e', 'Unknown': '#cccccc'},
            title='Language Distribution by Magazine',
            barmode='stack'
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title='Magazine',
            yaxis_title='Number of Articles',
            legend_title='Language',
            hovermode='closest'
        )
        
        # Display the interactive chart
        st.plotly_chart(fig, use_container_width=True)
    
    # Theme distribution chart
    st.subheader("Theme Distribution")
    
    # Prepare data for theme chart
    theme_counts = df['theme'].value_counts().reset_index()
    theme_counts.columns = ['Theme', 'Count']
    theme_counts = theme_counts.sort_values('Count', ascending=False)
    
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
    
    # Display the interactive chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Top authors chart
    st.subheader("Top 10 Authors")
    
    # Prepare data for authors chart
    author_counts = df['author'].value_counts().head(10).reset_index()
    author_counts.columns = ['Author', 'Count']
    
    # Create a horizontal bar chart using Plotly
    fig = px.bar(
        author_counts,
        y='Author',
        x='Count',
        color='Count',
        color_continuous_scale='Oranges',
        title='Top 10 Authors by Article Count',
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
    
    # Display the interactive chart
    st.plotly_chart(fig, use_container_width=True)

def show_validation(df: pd.DataFrame):
    """Display data validation checks and results"""
    st.title("🔍 Data Validation")
    
    # Overview of validation checks
    st.write("""
    This section runs data quality checks on the extracted metadata to identify potential issues.
    """)
    
    expected_ranges = {
        'WE': set(range(48, 64)),  # 48 to 63
        'Orizzonti': set(range(55, 65))  # 55 to 64
    }
    
    expected_counts = {
        'WE': 16,
        'Orizzonti': 12
    }
    
    # Run validation checks
    with st.spinner("Running validation checks..."):
        validation_results = run_data_validation(df, expected_ranges, expected_counts)
    
    # Show validation results in expandable sections
    with st.expander("Magazine Name Validation", expanded=True):
        st.subheader("Magazine Names")
        
        if validation_results['invalid_magazines']:
            st.error(f"Found {len(validation_results['invalid_magazines'])} invalid magazine names:")
            for mag in validation_results['invalid_magazines']:
                st.write(f"  - '{mag}'")
        else:
            st.success("✅ All magazine names are valid (Orizzonti or WE)")
    
    with st.expander("Issue Number Validation", expanded=True):
        st.subheader("Magazine Issue Numbers")
        
        # Report issues out of range
        if validation_results['out_of_range_issues']:
            st.error(f"Found {len(validation_results['out_of_range_issues'])} issues outside expected ranges:")
            for item in validation_results['out_of_range_issues']:
                st.write(f"  - {item['magazine']} issue {item['issue']} (normalized: {item['normalized_issue']})")
        else:
            st.success("✅ All magazine issues are within expected ranges")
        
        # Report missing issues
        missing_count = sum(len(issues) for issues in validation_results['missing_issues'].values())
        if missing_count > 0:
            st.warning(f"Missing {missing_count} expected issues:")
            for magazine, issues in validation_results['missing_issues'].items():
                if issues:
                    st.write(f"  - {magazine}: missing issues {', '.join(map(str, issues))}")
        else:
            st.success("✅ No missing issues detected")
    
    with st.expander("Article Count Validation", expanded=True):
        st.subheader("Article Counts per Issue")
        
        # Report issues with low article counts
        if validation_results['issues_with_low_articles']:
            st.warning(f"Found {len(validation_results['issues_with_low_articles'])} issues with fewer than 50% of expected articles:")
            for item in validation_results['issues_with_low_articles']:
                st.write(f"  - {item['magazine']} {item['issue']}: {item['count']} articles (only {item['percentage']}% of expected {item['expected']})")
        else:
            st.success("✅ All issues have reasonable article counts")
    
    with st.expander("Field Validation", expanded=True):
        st.subheader("Required Fields")
        
        # Report missing fields
        if validation_results['missing_fields']:
            st.error(f"Fields missing from dataset: {', '.join(validation_results['missing_fields'].keys())}")
        else:
            st.success("✅ All required fields exist in the dataset")
        
        # Report null values
        if validation_results['null_counts']:
            st.warning(f"Found null values in these fields:")
            for field, count in validation_results['null_counts'].items():
                percentage = round(count / len(df) * 100, 1)
                st.write(f"  - {field}: {count} nulls ({percentage}%)")
        else:
            st.success("✅ No null values found in required fields")
        
        # Report empty strings
        if validation_results['empty_fields']:
            st.warning(f"Found empty strings in these fields:")
            for field, count in validation_results['empty_fields'].items():
                percentage = round(count / len(df) * 100, 1)
                st.write(f"  - {field}: {count} empty strings ({percentage}%)")
        else:
            st.success("✅ No empty strings found in required fields")

def run_data_validation(df: pd.DataFrame, expected_ranges: Dict[str, Set[int]], expected_counts: Dict[str, int]) -> Dict[str, Any]:
    """Run all validation checks and return results"""
    results = {
        'invalid_magazines': [],
        'out_of_range_issues': [],
        'missing_issues': {},
        'issues_with_low_articles': [],
        'missing_fields': {},
        'empty_fields': {},
        'null_counts': {}
    }
    
    # Validate magazine names
    valid_magazines = {'Orizzonti', 'WE'}
    results['invalid_magazines'] = df[~df['magazine'].isin(valid_magazines)]['magazine'].unique().tolist()
    
    # Normalize magazine_no to integers
    df['magazine_no_norm'] = df['magazine_no'].apply(
        lambda x: int(float(x)) if pd.notnull(x) and str(x).replace('.', '', 1).isdigit() else None
    )
    
    # Validate issue numbers
    for magazine in expected_ranges:
        magazine_df = df[df['magazine'] == magazine]
        if len(magazine_df) == 0:
            continue
            
        for idx, row in magazine_df.iterrows():
            issue = row['magazine_no_norm']
            if issue is not None and issue not in expected_ranges[magazine]:
                results['out_of_range_issues'].append({
                    'magazine': magazine,
                    'issue': row['magazine_no'],
                    'normalized_issue': issue
                })
    
    # Find missing issues
    for magazine, expected_issues in expected_ranges.items():
        magazine_df = df[df['magazine'] == magazine]
        if len(magazine_df) == 0:
            results['missing_issues'][magazine] = list(expected_issues)
            continue
            
        found_issues = set(magazine_df['magazine_no_norm'].dropna().unique())
        missing_issues = expected_issues - found_issues
        if missing_issues:
            results['missing_issues'][magazine] = sorted(list(missing_issues))
    
    # Check article counts
    issue_counts = df.groupby(['magazine', 'magazine_no']).size()
    
    # Find issues with significantly fewer articles than expected
    threshold_percentage = 0.5  # 50% of expected
    
    for (magazine, issue), count in issue_counts.items():
        if magazine in expected_counts:
            expected = expected_counts[magazine]
            if count < expected * threshold_percentage:
                results['issues_with_low_articles'].append({
                    'magazine': magazine,
                    'issue': issue,
                    'count': count,
                    'expected': expected,
                    'percentage': round(count / expected * 100, 1)
                })
    
    # Validate required fields
    required_fields = [
        'author', 'title', 'magazine', 'magazine_no', 
        'abstract', 'theme', 'format', 'geographic_area', 'keywords'
    ]
    
    # Check for missing fields
    for field in required_fields:
        if field not in df.columns:
            results['missing_fields'][field] = True
    
    # Count nulls in each field
    for field in required_fields:
        if field in df.columns:
            null_count = df[field].isnull().sum()
            if null_count > 0:
                results['null_counts'][field] = null_count
    
    # Count empty strings in each field
    for field in required_fields:
        if field in df.columns and df[field].dtype == object:
            empty_count = (df[field] == "").sum()
            if empty_count > 0:
                results['empty_fields'][field] = empty_count
    
    return results

def show_explorer(df: pd.DataFrame):
    """Unified explorer that combines magazine and article browsing"""
    st.title("🔎 Content Explorer")
    
    # Filters in expander
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Magazine filter
            magazines = ['All'] + sorted(df['magazine'].unique().tolist())
            selected_magazine = st.selectbox("Magazine", magazines)
        
        with col2:
            # Issue number filter
            if selected_magazine == 'All':
                issues = ['All'] + sorted(df['magazine_no'].unique().tolist(), 
                                         key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf'))
            else:
                issues = ['All'] + sorted(df[df['magazine'] == selected_magazine]['magazine_no'].unique().tolist(),
                                         key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf'))
            selected_issue = st.selectbox("Issue", issues)
        
        with col3:
            # Theme filter
            themes = ['All'] + sorted(df['theme'].unique().tolist())
            selected_theme = st.selectbox("Theme", themes)
        
        # Additional filters row
        col1, col2 = st.columns(2)
        
        with col1:
            # Author filter
            authors = ['All'] + sorted(df['author'].unique().tolist())
            selected_author = st.selectbox("Author", authors)
        
        with col2:
            # Search by keyword or title
            search_term = st.text_input("Search by title, keywords, or content")
    
    # Apply filters
    filtered_df = df.copy()
    if selected_magazine != 'All':
        filtered_df = filtered_df[filtered_df['magazine'] == selected_magazine]
    if selected_issue != 'All':
        filtered_df = filtered_df[filtered_df['magazine_no'] == selected_issue]
    if selected_theme != 'All':
        filtered_df = filtered_df[filtered_df['theme'] == selected_theme]
    if selected_author != 'All':
        filtered_df = filtered_df[filtered_df['author'] == selected_author]
    if search_term:
        # Search in multiple columns
        search_mask = (
            filtered_df['title'].str.contains(search_term, case=False, na=False) | 
            filtered_df['keywords'].str.contains(search_term, case=False, na=False) | 
            filtered_df['abstract'].str.contains(search_term, case=False, na=False) |
            filtered_df['abstract_ita'].str.contains(search_term, case=False, na=False) |
            filtered_df['abstract_eng'].str.contains(search_term, case=False, na=False) |
            filtered_df['geographic_area'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[search_mask]
    
    # Display results count
    st.write(f"Found {len(filtered_df)} articles")
    
    # Display articles
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        with st.expander(f"{row['title']} ({row['magazine']} {row['magazine_no']})"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Try to display preview image if available
                if 'preview_image_path' in row and row['preview_image_path'] and row['preview_image_path'] != 'nan':
                    if os.path.exists(row['preview_image_path']):
                        st.image(row['preview_image_path'], width=200)
                    else:
                        st.info("Image not found")
            
            with col2:
                # Display metadata
                st.write(f"**Author:** {row['author']}")
                st.write(f"**Language:** {row.get('language', 'Not specified')}")
                
                # Display abstract based on language
                if 'language' in row and row['language'] == 'ITA' and 'abstract_ita' in row and row['abstract_ita']:
                    st.write(f"**Abstract (ITA):** {row['abstract_ita']}")
                    if 'abstract_eng' in row and row['abstract_eng']:
                        st.write(f"**Abstract (ENG):** {row['abstract_eng']}")
                elif 'language' in row and row['language'] == 'ENG' and 'abstract_eng' in row and row['abstract_eng']:
                    st.write(f"**Abstract (ENG):** {row['abstract_eng']}")
                    if 'abstract_ita' in row and row['abstract_ita']:
                        st.write(f"**Abstract (ITA):** {row['abstract_ita']}")
                else:
                    # Fallback to original abstract
                    st.write(f"**Abstract:** {row['abstract']}")
                
                st.write(f"**Theme:** {row['theme']}")
                st.write(f"**Format:** {row['format']}")
                st.write(f"**Geographic Area:** {row['geographic_area']}")
                st.write(f"**Keywords:** {row['keywords']}")

if __name__ == "__main__":
    setup_page()
    main() 