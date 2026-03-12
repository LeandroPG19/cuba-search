"""Cuba-Search MCP entry point for python -m cuba_search."""
import asyncio

from cuba_search.protocol import main

if __name__ == "__main__":
    asyncio.run(main())
