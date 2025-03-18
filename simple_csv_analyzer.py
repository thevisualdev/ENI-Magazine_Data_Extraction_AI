#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple, Any, Set

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

# Navigation
pages = [
    "Overview", 
    "Magazine Explorer",
    "Article Browser",
    "Data Validation",
    "Theme Analysis"
]

page = st.sidebar.radio("Select Page", pages)

# Basic stats
if page == "Overview":
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
    magazine_counts = df['magazine'].value_counts().reset_index()
    magazine_counts.columns = ['Magazine', 'Count']
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(magazine_counts['Magazine'], magazine_counts['Count'], color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Number of Articles')
    ax.set_title('Articles per Magazine')
    
    # Add labels
    for i, count in enumerate(magazine_counts['Count']):
        ax.text(i, count + 5, str(count), ha='center')
    
    st.pyplot(fig)
    
    # Articles per issue
    st.subheader("Articles per Issue")
    issue_counts = df.groupby(['magazine', 'magazine_no']).size().reset_index()
    issue_counts.columns = ['Magazine', 'Issue', 'Count']
    
    # Sort by magazine and issue
    issue_counts['Issue_Numeric'] = pd.to_numeric(issue_counts['Issue'], errors='coerce')
    issue_counts = issue_counts.sort_values(['Magazine', 'Issue_Numeric'])
    
    # Group by magazine
    magazines = issue_counts['Magazine'].unique()
    
    for magazine in magazines:
        st.write(f"### {magazine}")
        magazine_data = issue_counts[issue_counts['Magazine'] == magazine]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(magazine_data['Issue'], magazine_data['Count'], color='#1f77b4')
        
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
        
        st.pyplot(fig)
    
    # Top authors
    st.subheader("Top 10 Authors")
    top_authors = df['author'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_authors.index[::-1], top_authors.values[::-1], color='#1f77b4')
    ax.set_xlabel('Number of Articles')
    ax.set_title('Top 10 Authors by Article Count')
    
    # Add count labels
    for i, count in enumerate(top_authors.values[::-1]):
        ax.text(count + 0.2, i, str(count), va='center')
    
    st.pyplot(fig)
    
    # Theme distribution
    st.subheader("Theme Distribution")
    theme_counts = df['theme'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        theme_counts, 
        labels=theme_counts.index, 
        autopct='%1.1f%%',
        textprops={'fontsize': 9}
    )
    ax.axis('equal')
    plt.tight_layout()
    
    st.pyplot(fig)

elif page == "Magazine Explorer":
    st.header("Magazine Explorer")
    
    st.write("""
    Explore articles by magazine and issue number.
    """)
    
    # Magazine and issue selection
    col1, col2 = st.columns(2)
    
    with col1:
        magazines = sorted(df['magazine'].unique())
        selected_magazine = st.selectbox("Select Magazine", magazines)
        
    # Filter by magazine
    magazine_df = df[df['magazine'] == selected_magazine]
    
    with col2:
        issues = sorted(magazine_df['magazine_no'].unique(), 
                       key=lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else 0)
        selected_issue = st.selectbox("Select Issue", issues)
    
    # Filter by issue
    issue_df = magazine_df[magazine_df['magazine_no'] == selected_issue]
    
    # Display issue info
    st.subheader(f"{selected_magazine} - Issue {selected_issue}")
    st.write(f"Found {len(issue_df)} articles in this issue")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Articles", len(issue_df))
    with col2:
        st.metric("Unique Authors", issue_df['author'].nunique())
    with col3:
        st.metric("Themes Covered", issue_df['theme'].nunique())
    
    # Display articles
    st.subheader("Articles in this issue")
    
    for i, (_, article) in enumerate(issue_df.iterrows(), 1):
        with st.expander(f"{i}. {article['title']} (by {article['author']})"):
            # Display article details
            st.write(f"**Abstract:** {article['abstract']}")
            st.write(f"**Theme:** {article['theme']}")
            st.write(f"**Format:** {article['format']}")
            st.write(f"**Geographic Area:** {article['geographic_area']}")
            st.write(f"**Keywords:** {article['keywords']}")

elif page == "Article Browser":
    st.header("Article Browser")
    
    st.write("""
    Browse and search for specific articles.
    """)
    
    # Search and filter options
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.text_input("Search in Title, Abstract or Keywords")
    
    with col2:
        # Theme filter
        themes = ["All Themes"] + sorted(df['theme'].unique().tolist())
        selected_theme = st.selectbox("Filter by Theme", themes)
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply theme filter
    if selected_theme != "All Themes":
        filtered_df = filtered_df[filtered_df['theme'] == selected_theme]
    
    # Apply search filter
    if search_term:
        # Search in title, abstract and keywords
        title_match = filtered_df['title'].str.contains(search_term, case=False, na=False)
        abstract_match = filtered_df['abstract'].str.contains(search_term, case=False, na=False)
        keywords_match = filtered_df['keywords'].str.contains(search_term, case=False, na=False)
        
        filtered_df = filtered_df[title_match | abstract_match | keywords_match]
    
    # Display filtered results
    st.write(f"Found {len(filtered_df)} articles matching your criteria")
    
    # Display results in pages
    articles_per_page = 10
    num_pages = max(1, (len(filtered_df) + articles_per_page - 1) // articles_per_page)
    
    if len(filtered_df) > 0:
        page_num = st.number_input("Page", min_value=1, max_value=num_pages, value=1)
        
        # Calculate start and end indices
        start_idx = (page_num - 1) * articles_per_page
        end_idx = min(start_idx + articles_per_page, len(filtered_df))
        
        # Get articles for current page
        page_df = filtered_df.iloc[start_idx:end_idx]
        
        # Display articles
        for i, (_, article) in enumerate(page_df.iterrows(), start_idx + 1):
            with st.expander(f"{i}. {article['title']} ({article['magazine']} {article['magazine_no']})"):
                # Display article details
                st.write(f"**Author:** {article['author']}")
                st.write(f"**Abstract:** {article['abstract']}")
                st.write(f"**Theme:** {article['theme']}")
                st.write(f"**Format:** {article['format']}")
                st.write(f"**Geographic Area:** {article['geographic_area']}")
                st.write(f"**Keywords:** {article['keywords']}")
    else:
        st.info("No articles found matching your criteria.")

elif page == "Data Validation":
    st.header("Data Validation")
    
    st.write("""
    Check data quality and validate metadata.
    """)
    
    # Define expected ranges and values
    expected_ranges = {
        'WE': list(range(48, 64)),  # 48 to 63
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
                if issue is not None and issue not in expected_issues:
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

elif page == "Theme Analysis":
    st.header("Theme Analysis")
    
    st.write("""
    Analyze themes and their distribution across magazines and issues.
    """)
    
    # Overall theme distribution
    st.subheader("Overall Theme Distribution")
    
    theme_counts = df['theme'].value_counts()
    
    # Create a bar chart for themes
    fig, ax = plt.subplots(figsize=(12, 6))
    theme_counts.plot(kind='barh', ax=ax)
    ax.set_xlabel('Number of Articles')
    ax.set_title('Distribution of Themes')
    ax.invert_yaxis()  # Show highest count at the top
    
    # Add count labels
    for i, count in enumerate(theme_counts):
        ax.text(count + 1, i, str(count), va='center')
    
    st.pyplot(fig)
    
    # Themes by magazine
    st.subheader("Themes by Magazine")
    
    # Create a radio button to select magazine
    magazine = st.radio("Select Magazine", sorted(df['magazine'].unique()))
    
    # Filter to selected magazine
    magazine_df = df[df['magazine'] == magazine]
    
    # Calculate theme counts for this magazine
    magazine_theme_counts = magazine_df['theme'].value_counts()
    
    # Create a pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        magazine_theme_counts, 
        labels=magazine_theme_counts.index, 
        autopct='%1.1f%%',
        textprops={'fontsize': 9}
    )
    ax.axis('equal')
    ax.set_title(f'Theme Distribution in {magazine}')
    
    st.pyplot(fig)
    
    # Theme trends across issues
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
        # Filter to selected themes
        theme_df = magazine_df[magazine_df['theme'].isin(selected_themes)]
        
        # Group by issue and theme
        try:
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
        except:
            st.warning("Not enough data to create theme trends chart for the selected themes.")
    else:
        st.info("Please select at least one theme to display.") 