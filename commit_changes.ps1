# PowerShell script to commit changes

# Set Git user information
git config --global user.email "user@example.com"
git config --global user.name "User"

# Add all changes
git add .

# Commit with message
git commit -m "Update FastMCP version to 2.11.3 and fix logging configuration"

# Show status
git status
