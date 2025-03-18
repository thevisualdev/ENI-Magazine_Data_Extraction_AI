#!/bin/bash
#
# Script to remove API keys from Git history
# WARNING: This rewrites Git history. Only use if you understand the consequences.
#
# Usage: bash scrub_api_keys.sh

echo "WARNING: This script will rewrite Git history to remove API keys."
echo "This is a destructive operation that will change commit hashes."
echo "All collaborators will need to re-clone the repository after this."
echo ""
read -p "Are you sure you want to continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Operation cancelled."
    exit 1
fi

# Use BFG Repo-Cleaner if available
if command -v bfg &> /dev/null; then
    echo "BFG found, using it to clean the repository..."
    
    # Example for replacing API keys
    read -p "Enter your API key pattern (e.g., 'sk-[a-zA-Z0-9]{48}'): " apikey_pattern
    if [ -z "$apikey_pattern" ]; then
        apikey_pattern="sk-[a-zA-Z0-9]{48}"
        echo "Using default pattern: $apikey_pattern"
    fi
    
    # Create a text file with replacement strings
    echo "$apikey_pattern=***REMOVED-API-KEY***" > replace-tokens.txt
    
    # Run BFG to replace the strings
    bfg --replace-text replace-tokens.txt
    
    # Remove the replacement file
    rm replace-tokens.txt
    
else
    echo "BFG not found, using git filter-branch (slower)..."
    
    # Use git filter-branch to remove the API key
    git filter-branch --force --index-filter \
        'git ls-files -z "*.yaml" "*.yml" "*.json" "*.py" | xargs -0 sed -i "" "s/sk-[a-zA-Z0-9]\{48\}/***REMOVED-API-KEY***/g"' \
        --prune-empty --tag-name-filter cat -- --all
fi

echo ""
echo "Clean up temporary files and force garbage collection..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Git history has been rewritten."
echo "Next steps:"
echo "1. Force push to all branches: git push --force --all"
echo "2. Force push tags: git push --force --tags"
echo "3. Inform all collaborators to re-clone the repository" 