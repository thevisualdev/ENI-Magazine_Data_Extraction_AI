#!/usr/bin/env python3
import pandas as pd
import os
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Set

"""
Validation script to check the quality of extracted magazine data
"""

def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame"""
    print(f"Loading CSV from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records")
    return df

def analyze_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic statistics about the dataset"""
    stats = {}
    
    # Count records by magazine
    magazine_counts = df['magazine'].value_counts().to_dict()
    stats['magazine_counts'] = magazine_counts
    print(f"\nMagazine distribution:")
    for magazine, count in magazine_counts.items():
        print(f"- {magazine}: {count} articles")
    
    # Count records by magazine issue
    magazine_issue_counts = df.groupby(['magazine', 'magazine_no']).size().to_dict()
    stats['magazine_issue_counts'] = magazine_issue_counts
    print(f"\nArticles per magazine issue:")
    for (magazine, issue), count in sorted(magazine_issue_counts.items()):
        print(f"- {magazine} {issue}: {count} articles")
    
    # Top authors
    top_authors = df['author'].value_counts().head(10).to_dict()
    stats['top_authors'] = top_authors
    print(f"\nTop 10 authors:")
    for author, count in top_authors.items():
        print(f"- {author}: {count} articles")
    
    # Top themes
    theme_counts = df['theme'].value_counts().to_dict()
    stats['theme_counts'] = theme_counts
    print(f"\nTheme distribution:")
    for theme, count in theme_counts.items():
        print(f"- {theme}: {count} articles")
    
    # Count formats
    format_counts = df['format'].value_counts().to_dict()
    stats['format_counts'] = format_counts
    print(f"\nFormat distribution:")
    for format_type, count in format_counts.items():
        print(f"- {format_type}: {count} articles")
    
    return stats

def validate_magazine_names(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Check if all magazine names are valid (Orizzonti or WE)"""
    valid_magazines = {'Orizzonti', 'WE'}
    invalid_magazines = df[~df['magazine'].isin(valid_magazines)]['magazine'].unique().tolist()
    
    print(f"\nValidating magazine names:")
    if invalid_magazines:
        print(f"❌ Found {len(invalid_magazines)} invalid magazine names:")
        for mag in invalid_magazines:
            print(f"  - '{mag}'")
    else:
        print(f"✅ All magazine names are valid (Orizzonti or WE)")
    
    return {'invalid_magazines': invalid_magazines}

def validate_magazine_numbers(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check if magazine numbers are in the expected range:
    - WE: 48-63
    - Orizzonti: 55-64
    """
    expected_ranges = {
        'WE': set(range(48, 64)),  # 48 to 63
        'Orizzonti': set(range(55, 65))  # 55 to 64
    }
    
    results = {
        'out_of_range_issues': [],
        'missing_issues': {}
    }
    
    # First normalize magazine_no to integers
    df['magazine_no_norm'] = df['magazine_no'].apply(
        lambda x: int(float(x)) if pd.notnull(x) and str(x).replace('.', '', 1).isdigit() else None
    )
    
    # Find issues out of expected range
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
    
    print(f"\nValidating magazine issue numbers:")
    
    # Report issues out of range
    if results['out_of_range_issues']:
        print(f"❌ Found {len(results['out_of_range_issues'])} issues outside expected ranges:")
        for item in results['out_of_range_issues']:
            print(f"  - {item['magazine']} issue {item['issue']} (normalized: {item['normalized_issue']})")
    else:
        print(f"✅ All magazine issues are within expected ranges")
    
    # Report missing issues
    missing_count = sum(len(issues) for issues in results['missing_issues'].values())
    if missing_count > 0:
        print(f"⚠️ Missing {missing_count} expected issues:")
        for magazine, issues in results['missing_issues'].items():
            if issues:
                print(f"  - {magazine}: missing issues {', '.join(map(str, issues))}")
    else:
        print(f"✅ No missing issues detected")
    
    return results

def check_article_counts(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check if article counts per issue match expected values:
    - WE: ~16 articles per issue
    - Orizzonti: ~12 articles per issue
    """
    expected_counts = {
        'WE': 16,
        'Orizzonti': 12
    }
    
    stats = {
        'issue_article_counts': {},
        'issues_with_low_articles': []
    }
    
    # Count articles per issue
    issue_counts = df.groupby(['magazine', 'magazine_no']).size()
    stats['issue_article_counts'] = issue_counts.to_dict()
    
    # Find issues with significantly fewer articles than expected
    threshold_percentage = 0.5  # 50% of expected
    
    for (magazine, issue), count in issue_counts.items():
        if magazine in expected_counts:
            expected = expected_counts[magazine]
            if count < expected * threshold_percentage:
                stats['issues_with_low_articles'].append({
                    'magazine': magazine,
                    'issue': issue,
                    'count': count,
                    'expected': expected,
                    'percentage': round(count / expected * 100, 1)
                })
    
    print(f"\nChecking article counts per issue:")
    if stats['issues_with_low_articles']:
        print(f"⚠️ Found {len(stats['issues_with_low_articles'])} issues with fewer than {threshold_percentage*100}% of expected articles:")
        for item in stats['issues_with_low_articles']:
            print(f"  - {item['magazine']} {item['issue']}: {item['count']} articles (only {item['percentage']}% of expected {item['expected']})")
    else:
        print(f"✅ All issues have reasonable article counts")
    
    return stats

def validate_required_fields(df: pd.DataFrame) -> Dict[str, Any]:
    """Check if all required fields are populated"""
    required_fields = [
        'author', 'title', 'magazine', 'magazine_no', 
        'abstract', 'theme', 'format', 'geographic_area', 'keywords'
    ]
    
    stats = {
        'missing_fields': {},
        'empty_fields': {},
        'null_counts': {}
    }
    
    # Check for missing fields
    for field in required_fields:
        if field not in df.columns:
            stats['missing_fields'][field] = True
    
    # Count nulls in each field
    for field in required_fields:
        if field in df.columns:
            null_count = df[field].isnull().sum()
            if null_count > 0:
                stats['null_counts'][field] = null_count
    
    # Count empty strings in each field
    for field in required_fields:
        if field in df.columns and df[field].dtype == object:
            empty_count = (df[field] == "").sum()
            if empty_count > 0:
                stats['empty_fields'][field] = empty_count
    
    print(f"\nValidating required fields:")
    
    # Report missing fields
    if stats['missing_fields']:
        print(f"❌ Fields missing from dataset: {', '.join(stats['missing_fields'].keys())}")
    else:
        print(f"✅ All required fields exist in the dataset")
    
    # Report null values
    if stats['null_counts']:
        print(f"⚠️ Found null values in these fields:")
        for field, count in stats['null_counts'].items():
            percentage = round(count / len(df) * 100, 1)
            print(f"  - {field}: {count} nulls ({percentage}%)")
    else:
        print(f"✅ No null values found in required fields")
    
    # Report empty strings
    if stats['empty_fields']:
        print(f"⚠️ Found empty strings in these fields:")
        for field, count in stats['empty_fields'].items():
            percentage = round(count / len(df) * 100, 1)
            print(f"  - {field}: {count} empty strings ({percentage}%)")
    else:
        print(f"✅ No empty strings found in required fields")
    
    return stats

def show_sample_articles(df: pd.DataFrame, n: int = 5) -> None:
    """Display a sample of articles for manual inspection"""
    print(f"\nRandom sample of {n} articles for manual inspection:")
    sample = df.sample(n)
    
    for i, (_, article) in enumerate(sample.iterrows(), 1):
        print(f"\nSample article {i}:")
        print(f"  Title: {article.get('title', 'N/A')}")
        print(f"  Magazine: {article.get('magazine', 'N/A')} {article.get('magazine_no', 'N/A')}")
        print(f"  Author: {article.get('author', 'N/A')}")
        print(f"  Abstract: {article.get('abstract', 'N/A')[:100]}...")
        print(f"  Theme: {article.get('theme', 'N/A')}")
        print(f"  Format: {article.get('format', 'N/A')}")
        print(f"  Geographic Area: {article.get('geographic_area', 'N/A')}")
        print(f"  Keywords: {article.get('keywords', 'N/A')}")

def plot_visualizations(df: pd.DataFrame, output_dir: str = "output/validation") -> None:
    """Generate visualization plots for the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bar chart of articles per magazine
    plt.figure(figsize=(10, 6))
    df['magazine'].value_counts().plot(kind='bar')
    plt.title('Number of Articles per Magazine')
    plt.xlabel('Magazine')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/articles_per_magazine.png")
    
    # 2. Bar chart of articles per theme
    plt.figure(figsize=(12, 6))
    theme_counts = df['theme'].value_counts()
    theme_counts.plot(kind='bar')
    plt.title('Distribution of Articles by Theme')
    plt.xlabel('Theme')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/articles_by_theme.png")
    
    # 3. Pie chart of article formats
    plt.figure(figsize=(10, 8))
    df['format'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Article Formats')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/article_formats.png")
    
    # 4. Articles per issue for each magazine
    plt.figure(figsize=(14, 6))
    issue_counts = df.groupby(['magazine', 'magazine_no']).size().reset_index(name='count')
    
    # Filter to just numeric issue numbers for proper plotting
    issue_counts['magazine_no_numeric'] = pd.to_numeric(issue_counts['magazine_no'], errors='coerce')
    issue_counts = issue_counts.dropna(subset=['magazine_no_numeric'])
    
    # Sort by magazine and issue number
    issue_counts = issue_counts.sort_values(['magazine', 'magazine_no_numeric'])
    
    # Plot for each magazine
    for magazine in issue_counts['magazine'].unique():
        magazine_data = issue_counts[issue_counts['magazine'] == magazine]
        plt.plot(magazine_data['magazine_no_numeric'], magazine_data['count'], 
                 marker='o', label=magazine)
    
    plt.title('Number of Articles per Issue')
    plt.xlabel('Issue Number')
    plt.ylabel('Number of Articles')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/articles_per_issue.png")
    
    print(f"\nPlots saved to {output_dir}/")

def main():
    """Main function to run all validations"""
    csv_path = "output/extracted_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
    
    # Load the data
    df = load_csv(csv_path)
    
    # Run validations
    print("\n" + "="*50)
    print("VALIDATING CSV DATA QUALITY")
    print("="*50)
    
    # Basic statistics
    analyze_basic_stats(df)
    
    # Validate magazine names
    validate_magazine_names(df)
    
    # Validate magazine numbers
    validate_magazine_numbers(df)
    
    # Check article counts
    check_article_counts(df)
    
    # Validate required fields
    validate_required_fields(df)
    
    # Show sample articles
    show_sample_articles(df, n=3)
    
    # Generate plots for visualization
    plot_visualizations(df)
    
    print("\n" + "="*50)
    print("VALIDATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main() 