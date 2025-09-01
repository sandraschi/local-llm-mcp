from setuptools import setup, find_packages

setup(
    name="llm-mcp",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastmcp>=2.11.3",  # Stable version supporting required functionality
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.9.0",
        "openai>=1.12.0",
        "anthropic>=0.13.0",
        "vllm>=1.0.0",
        "transformers>=4.36.0",
        "sentence-transformers>=2.2.2",
        "Pillow>=10.0.0",
        "psutil>=5.9.6",
        "GPUtil>=1.4.2",
        "typing-extensions>=4.8.0",
        "python-dateutil>=2.8.2",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "llm-mcp=llm_mcp.main:main",
        ],
    },
)
