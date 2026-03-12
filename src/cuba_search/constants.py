"""Constants, tool definitions, and configuration for cuba-search."""

# ── Tool Definitions (MCP Schema) ──────────────────────────────────
TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "cuba_search",
        "description": (
            "Search the web for current information on any topic. "
            "Returns ranked, deduplicated results with source credibility tiers. "
            "Supports time filtering, language, and category selection."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results (default 10, max 20)",
                },
                "categories": {
                    "type": "string",
                    "description": "Category: general, news, science, it, images, videos",
                },
                "language": {
                    "type": "string",
                    "description": "Language code (auto, en, es, etc.)",
                },
                "time_range": {
                    "type": "string",
                    "description": "Time filter: day, week, month, year",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "cuba_scrape",
        "description": (
            "Scrape a single URL and extract clean, readable content. "
            "Uses readability algorithm. Respects robots.txt. SSRF-protected."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to scrape",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Max tokens in response (default 2000)",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "cuba_crawl",
        "description": (
            "Crawl a website from a starting URL. Extracts content from pages "
            "with configurable depth. Same-domain only. Respects robots.txt."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Starting URL",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Max pages to crawl (default 10, max 20)",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Max depth from start (default 2, max 3)",
                },
                "instructions": {
                    "type": "string",
                    "description": "Filter: only return pages matching these keywords",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "cuba_extract",
        "description": (
            "Extract structured content from one or more URLs. "
            "Returns clean text optimized for AI consumption."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URLs to extract content from",
                },
                "query": {
                    "type": "string",
                    "description": "Query to prioritize relevant content",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Max total tokens (default 3000)",
                },
            },
            "required": ["urls"],
        },
    },
    {
        "name": "cuba_map",
        "description": (
            "Discover URLs from a starting point. Returns list of URLs "
            "found on the page. Useful for sitemap discovery."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to discover links from",
                },
                "max_urls": {
                    "type": "integer",
                    "description": "Maximum URLs to return (default 50)",
                },
                "same_domain": {
                    "type": "boolean",
                    "description": "Only return same-domain URLs (default true)",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "cuba_validate",
        "description": (
            "Validate information by cross-referencing multiple sources. "
            "Returns confidence score, contradiction markers, and claim density."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "Claim or statement to validate",
                },
                "max_sources": {
                    "type": "integer",
                    "description": "Number of sources to check (default 5)",
                },
            },
            "required": ["claim"],
        },
    },
    {
        "name": "cuba_docs",
        "description": (
            "Query documentation for any library. Single-call replacement for "
            "Context7. Resolves library → official docs → scrapes → compresses. "
            "Self-hosted, real-time, no rate limits."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "library": {
                    "type": "string",
                    "description": "Library name (e.g., 'fastapi', 'react', 'sqlalchemy')",
                },
                "query": {
                    "type": "string",
                    "description": "What to search for in the docs",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Max response tokens (default 1500)",
                },
            },
            "required": ["library", "query"],
        },
    },
    {
        "name": "cuba_research",
        "description": (
            "Deep research on a topic: search → scrape → validate → compress. "
            "Combines all pipeline stages for comprehensive investigation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Research question or topic",
                },
                "depth": {
                    "type": "string",
                    "enum": ["quick", "standard", "deep"],
                    "description": "Research depth (default: standard)",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Max response tokens (default 3000)",
                },
            },
            "required": ["query"],
        },
    },
]
