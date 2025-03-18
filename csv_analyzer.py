#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple, Any, Set

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
    
    # Sidebar navigation
    nav_options = [
        "📈 Overview",
        "🔍 Data Validation", 
        "🔎 Magazine Explorer",
        "📚 Article Browser",
        "📊 Theme Analysis"
    ]
    
    nav_selection = st.sidebar.radio("Navigation", nav_options)
    
    # Display content based on selection
    if nav_selection == "📈 Overview":
        show_overview(df)
    elif nav_selection == "🔍 Data Validation":
        show_validation(df)
    elif nav_selection == "🔎 Magazine Explorer":
        show_magazine_explorer(df)
    elif nav_selection == "📚 Article Browser":
        show_article_browser(df)
    elif nav_selection == "📊 Theme Analysis":
        show_theme_analysis(df)

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
        st.metric("Formats", df['format'].nunique())
    
    # Articles per magazine chart
    st.subheader("Articles per Magazine")
    
    # Prepare data for chart
    magazine_counts = df['magazine'].value_counts().reset_index()
    magazine_counts.columns = ['Magazine', 'Count']
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(magazine_counts['Magazine'], magazine_counts['Count'], color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Number of Articles')
    ax.set_title('Articles per Magazine')
    
    # Customize the chart
    for i, count in enumerate(magazine_counts['Count']):
        ax.text(i, count + 5, str(count), ha='center')
    
    st.pyplot(fig)
    
    # Articles per issue chart
    st.subheader("Articles per Issue")
    
    # Prepare data for chart
    issue_counts = df.groupby(['magazine', 'magazine_no']).size().reset_index()
    issue_counts.columns = ['Magazine', 'Issue', 'Count']
    
    # Sort by magazine and issue number
    issue_counts['Issue_Numeric'] = pd.to_numeric(issue_counts['Issue'], errors='coerce')
    issue_counts = issue_counts.sort_values(['Magazine', 'Issue_Numeric'])
    
    # Create a grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot separately for each magazine with different colors
    magazines = issue_counts['Magazine'].unique()
    colors = ['#1f77b4', '#ff7f0e']
    
    for i, magazine in enumerate(magazines):
        data = issue_counts[issue_counts['Magazine'] == magazine]
        bar_positions = np.arange(len(data)) + i*0.4
        bars = ax.bar(bar_positions, data['Count'], width=0.4, label=magazine, color=colors[i % len(colors)])
        
        # Add issue labels
        for j, (_, row) in enumerate(data.iterrows()):
            ax.text(bar_positions[j], 0.5, str(row['Issue']), rotation=90, ha='center', va='bottom', fontsize=8)
    
    # Customize the chart
    ax.set_ylabel('Number of Articles')
    ax.set_title('Articles per Issue')
    ax.set_xticks([])
    ax.legend()
    
    st.pyplot(fig)
    
    # Top 10 authors
    st.subheader("Top 10 Authors")
    
    # Calculate top authors
    top_authors = df['author'].value_counts().head(10).reset_index()
    top_authors.columns = ['Author', 'Count']
    
    # Create a horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_authors['Author'], top_authors['Count'], color='#1f77b4')
    ax.set_xlabel('Number of Articles')
    ax.set_title('Top 10 Authors by Article Count')
    
    # Add count labels
    for i, count in enumerate(top_authors['Count']):
        ax.text(count + 0.2, i, str(count), va='center')
    
    # Reverse y-axis to show highest count at the top
    ax.invert_yaxis()
    
    st.pyplot(fig)
    
    # Theme distribution
    st.subheader("Theme Distribution")
    
    # Calculate theme counts
    theme_counts = df['theme'].value_counts().reset_index()
    theme_counts.columns = ['Theme', 'Count']
    
    # Create a pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        theme_counts['Count'], 
        labels=theme_counts['Theme'], 
        autopct='%1.1f%%',
        textprops={'fontsize': 9}
    )
    
    # Equal aspect ratio ensures the pie chart is circular
    ax.axis('equal')
    ax.set_title('Distribution of Articles by Theme')
    
    # Make room for the labels
    plt.tight_layout()
    
    st.pyplot(fig)

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

def show_magazine_explorer(df: pd.DataFrame):
    """Display interactive exploration by magazine and issue"""
    st.title("🔎 Magazine Explorer")
    
    st.write("""
    Explore magazine contents by selecting a specific magazine and issue number.
    You can see all articles for the selected issue and analyze their metadata.
    """)
    
    # Create filters for magazine and issue
    col1, col2 = st.columns(2)
    
    with col1:
        magazines = sorted(df['magazine'].unique())
        selected_magazine = st.selectbox("Select Magazine", magazines)
    
    # Filter issues based on selected magazine
    filtered_df = df[df['magazine'] == selected_magazine]
    issues = sorted(filtered_df['magazine_no'].unique(), key=lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else 0)
    
    with col2:
        selected_issue = st.selectbox("Select Issue", issues)
    
    # Filter to selected issue
    issue_df = filtered_df[filtered_df['magazine_no'] == selected_issue]
    
    # Display issue information
    st.subheader(f"{selected_magazine} - Issue {selected_issue}")
    st.write(f"Found {len(issue_df)} articles in this issue.")
    
    # Display articles in this issue
    st.write("### Articles in this issue")
    
    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Articles", len(issue_df))
    with col2:
        st.metric("Unique Authors", issue_df['author'].nunique())
    with col3:
        unique_themes = issue_df['theme'].nunique()
        st.metric("Themes Covered", unique_themes)
    
    # Display article list
    for i, (_, article) in enumerate(issue_df.iterrows(), 1):
        with st.expander(f"{i}. {article['title']} (by {article['author']})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Abstract:**")
                st.write(article['abstract'])
                
                st.write("**Keywords:**")
                st.write(article['keywords'])
            
            with col2:
                st.write("**Theme:**", article['theme'])
                st.write("**Format:**", article['format'])
                st.write("**Geographic Area:**", article['geographic_area'])
    
    # Display theme distribution for this issue
    st.subheader("Theme Distribution in this Issue")
    
    theme_counts = issue_df['theme'].value_counts()
    
    # Create a pie chart for themes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Only create the pie chart if we have multiple themes
    if len(theme_counts) > 1:
        wedges, texts, autotexts = ax.pie(
            theme_counts, 
            labels=theme_counts.index, 
            autopct='%1.1f%%',
            textprops={'fontsize': 9}
        )
        ax.axis('equal')
        ax.set_title('Theme Distribution')
        st.pyplot(fig)
    else:
        st.write(f"All articles share the same theme: {theme_counts.index[0]}")

def show_article_browser(df: pd.DataFrame):
    """Display interactive article browser with search and filters"""
    st.title("📚 Article Browser")
    
    st.write("""
    Browse and search all articles in the dataset. 
    Use the filters and search to find specific articles of interest.
    """)
    
    # Create sidebar filters
    st.sidebar.subheader("Filters")
    
    # Magazine filter
    magazines = sorted(df['magazine'].unique())
    selected_magazines = st.sidebar.multiselect("Magazine", magazines, default=magazines)
    
    # Theme filter
    themes = sorted(df['theme'].unique())
    selected_themes = st.sidebar.multiselect("Theme", themes)
    
    # Format filter
    formats = sorted(df['format'].unique())
    selected_formats = st.sidebar.multiselect("Format", formats)
    
    # Search box
    search_query = st.sidebar.text_input("Search (title, abstract, keywords)")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_magazines:
        filtered_df = filtered_df[filtered_df['magazine'].isin(selected_magazines)]
    
    if selected_themes:
        filtered_df = filtered_df[filtered_df['theme'].isin(selected_themes)]
        
    if selected_formats:
        filtered_df = filtered_df[filtered_df['format'].isin(selected_formats)]
    
    # Apply search if provided
    if search_query:
        # Search in title, abstract, and keywords
        title_match = filtered_df['title'].str.contains(search_query, case=False, na=False)
        abstract_match = filtered_df['abstract'].str.contains(search_query, case=False, na=False)
        keywords_match = filtered_df['keywords'].str.contains(search_query, case=False, na=False)
        
        filtered_df = filtered_df[title_match | abstract_match | keywords_match]
    
    # Display results
    st.write(f"Found {len(filtered_df)} articles matching your criteria.")
    
    # Show sort options
    sort_options = {
        "Magazine & Issue": lambda x: (x['magazine'], x['magazine_no_numeric']),
        "Title": lambda x: x['title'],
        "Author": lambda x: x['author'],
        "Theme": lambda x: x['theme']
    }
    
    sort_by = st.selectbox("Sort by", list(sort_options.keys()))
    
    # Apply sorting
    filtered_df = filtered_df.sort_values(by=sort_options[sort_by])
    
    # Display articles in pages
    articles_per_page = 10
    total_pages = (len(filtered_df) + articles_per_page - 1) // articles_per_page
    
    if total_pages > 0:
        # Create page selection
        if total_pages > 1:
            page = st.selectbox("Page", list(range(1, total_pages + 1)))
        else:
            page = 1
        
        # Calculate start and end indices
        start_idx = (page - 1) * articles_per_page
        end_idx = min(start_idx + articles_per_page, len(filtered_df))
        
        # Display articles for current page
        page_df = filtered_df.iloc[start_idx:end_idx]
        
        for i, (_, article) in enumerate(page_df.iterrows(), start_idx + 1):
            with st.expander(f"{i}. {article['title']} ({article['magazine']} {article['magazine_no']})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Author:** {article['author']}")
                    st.write("**Abstract:**")
                    st.write(article['abstract'])
                    
                    st.write("**Keywords:**")
                    st.write(article['keywords'])
                
                with col2:
                    st.write("**Theme:**", article['theme'])
                    st.write("**Format:**", article['format'])
                    st.write("**Geographic Area:**", article['geographic_area'])
                    
                    # Display file path
                    if 'file_path' in article:
                        st.write("**File Path:**")
                        st.text(os.path.basename(article['file_path']))
    else:
        st.warning("No articles found matching your criteria.")

def show_theme_analysis(df: pd.DataFrame):
    """Display theme-based analysis with trend charts"""
    st.title("📊 Theme Analysis")
    
    st.write("""
    This section analyzes the distribution of themes across magazines and issues,
    showing trends and patterns in the content.
    """)
    
    # Top themes overall
    st.subheader("Top Themes Overall")
    
    # Calculate theme counts
    theme_counts = df['theme'].value_counts()
    
    # Create a horizontal bar chart for themes
    fig, ax = plt.subplots(figsize=(10, 8))
    theme_counts.plot(kind='barh', ax=ax)
    ax.set_xlabel('Number of Articles')
    ax.set_title('Distribution of Themes')
    ax.invert_yaxis()  # Invert to show highest count at the top
    
    # Add count labels
    for i, count in enumerate(theme_counts):
        ax.text(count + 0.5, i, str(count), va='center')
    
    st.pyplot(fig)
    
    # Theme trends over time for each magazine
    st.subheader("Theme Trends Across Issues")
    
    # Create radio button to select magazine
    magazine = st.radio("Select Magazine", sorted(df['magazine'].unique()))
    
    # Filter to selected magazine
    magazine_df = df[df['magazine'] == magazine]
    
    # Create heatmap of themes across issues
    st.write(f"### Theme Distribution for {magazine}")
    
    # Get top 5 themes for this magazine
    top_themes = magazine_df['theme'].value_counts().head(5).index.tolist()
    selected_themes = st.multiselect("Select Themes to Display", 
                                   sorted(magazine_df['theme'].unique()), 
                                   default=top_themes)
    
    if selected_themes:
        # Filter to selected themes
        theme_df = magazine_df[magazine_df['theme'].isin(selected_themes)]
        
        # Group by issue and theme, count articles
        theme_issue_counts = theme_df.groupby(['magazine_no', 'theme']).size().unstack().fillna(0)
        
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
        
        st.pyplot(fig)
    else:
        st.warning("Please select at least one theme to display.")
    
    # Top authors by theme
    st.subheader("Top Authors by Theme")
    
    # Let user select a theme
    selected_theme = st.selectbox("Select Theme", sorted(df['theme'].unique()))
    
    # Filter to the selected theme
    theme_df = df[df['theme'] == selected_theme]
    
    # Count authors
    author_counts = theme_df['author'].value_counts().head(10)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    author_counts.plot(kind='barh', ax=ax)
    ax.set_xlabel('Number of Articles')
    ax.set_title(f'Top Authors for Theme: {selected_theme}')
    ax.invert_yaxis()  # Invert to show highest count at the top
    
    # Add count labels
    for i, count in enumerate(author_counts):
        ax.text(count + 0.2, i, str(count), va='center')
    
    st.pyplot(fig)
    
    # Geographic distribution by theme
    st.subheader("Geographic Focus by Theme")
    
    # Get geographic areas used more than once
    geo_counts = df.groupby(['theme', 'geographic_area']).size().reset_index()
    geo_counts.columns = ['Theme', 'Geographic_Area', 'Count']
    geo_counts = geo_counts[geo_counts['Count'] > 1]  # Only include counts > 1
    
    # Only include non-empty and not "Not Specified" geographic areas
    geo_counts = geo_counts[
        (geo_counts['Geographic_Area'].notna()) & 
        (geo_counts['Geographic_Area'] != '') &
        (geo_counts['Geographic_Area'] != 'Not Specified')
    ]
    
    # Get top geographic areas
    top_areas = geo_counts.groupby('Geographic_Area')['Count'].sum().nlargest(10).index.tolist()
    
    # Filter to only include top areas
    geo_counts = geo_counts[geo_counts['Geographic_Area'].isin(top_areas)]
    
    # Create a pivot table
    pivot = geo_counts.pivot_table(index='Theme', columns='Geographic_Area', values='Count', fill_value=0)
    
    # Sort by total count
    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('total', ascending=False)
    pivot = pivot.drop(columns=['total'])
    
    # Create a heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if not pivot.empty:
        im = ax.imshow(pivot, cmap='YlOrRd')
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        
        # Set tick labels
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticklabels(pivot.index)
        
        # Loop over data to create text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                if pivot.iloc[i, j] > 0:
                    ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", 
                            color="black" if pivot.iloc[i, j] < pivot.values.max() * 0.7 else "white")
        
        ax.set_title('Geographic Focus by Theme')
        plt.colorbar(im, ax=ax, label='Number of Articles')
        plt.tight_layout()
        
        st.pyplot(fig)
    else:
        st.warning("Not enough geographic data to create a meaningful visualization.")

if __name__ == "__main__":
    setup_page()
    main() 