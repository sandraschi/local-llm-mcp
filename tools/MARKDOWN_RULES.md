# Strict Markdown Formatting Rules

All Markdown files must follow these strict formatting rules:

## 1. Headers
- Use ATX-style headers with space after `#`
- Add one blank line before and after headers
- Use proper header hierarchy (one `#` for main title, `##` for sections, etc.)

```markdown
# Main Title

## Section

### Subsection
```

## 2. Lists
- Use `-` for unordered lists
- Add one blank line before and after lists
- Use 2 spaces for nested items
- Add one blank line between list items when they contain multiple paragraphs

```markdown
- First item
- Second item
  - Nested item
  - Another nested item
- Third item

1. First item
2. Second item
   - Nested bullet
   - Another nested bullet
3. Third item
```

## 3. Code Blocks
- Use triple backticks with language specification
- Add one blank line before and after code blocks

```python
def example():
    return "Properly formatted code block"
```

## 4. Links and Images
- Use descriptive link text
- Place the URL reference at the bottom of the document if using reference-style links

```markdown
[Link text](https://example.com)
![Alt text](image.jpg)
```

## 5. Tables
- Use pipes and dashes to create tables
- Ensure pipes align in the header separator
- Add one blank line before and after tables

```markdown
| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |
```

## 6. Blockquotes
- Use `>` for blockquotes
- Add one blank line before and after blockquotes

```markdown
> This is a blockquote.
> It can span multiple lines.
```

## 7. Horizontal Rules
- Use `---` for horizontal rules
- Add one blank line before and after

```markdown
---
```

## 8. Line Length
- Keep lines under 100 characters where possible
- Break long lines at appropriate points
- Use proper indentation for wrapped lines

## 9. Spacing
- Use one blank line between block-level elements
- No trailing whitespace
- End file with a single newline character

## 10. Emphasis
- Use `**bold**` for bold text
- Use `*italic*` for italic text
- Use `` `code` `` for inline code

## Example File Structure

```markdown
# Document Title

## Introduction

This is the introduction paragraph.

## Main Content

- First point
- Second point
  - Sub-point
  - Another sub-point

### Code Example

```python
def hello():
    print("Hello, World!")
```

## Conclusion

Final thoughts and summary.
```

These rules must be followed for all new Markdown files to ensure consistency and readability.
