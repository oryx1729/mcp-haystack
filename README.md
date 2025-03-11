# MCP Haystack Integration

This repository provides an integration between [Haystack](https://haystack.deepset.ai/) and the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction). MCP is an open protocol that standardizes how applications provide context to LLMs, similar to how USB-C provides a standardized way to connect devices.

## Overview

The MCP Haystack integration allows you to use MCP-compatible tools within Haystack pipelines. This enables your LLM applications to interact with external services and APIs through a standardized protocol, making it easier to build powerful, context-aware applications.

This integration includes a demo with the Rijksmuseum API, showcasing how to use MCP to search for artworks, get detailed information, and interact with the museum's collection.

## Installation

### Install from Source

```bash
git clone https://github.com/oryx1729/mcp-haystack.git
cd mcp-haystack
pip install -e .
```


## Docker

This repository includes a Dockerfile that sets up a complete environment with an MCP Server Demo.

### Building the Docker Image

```bash
docker build -t mcp-haystack-demo .
```

### Running the Docker Container

```bash
docker run -p 8888:8888 mcp-haystack-demo
```

## Rijksmuseum Demo

This repository includes a demo notebook that showcases how to use the MCP Haystack integration with the Rijksmuseum API. The demo allows you to:

- Search for artworks in the Rijksmuseum collection
- Get detailed information about specific artworks
- View artwork images
- Explore collections created by Rijksstudio users
- Get a chronological timeline of an artist's works

To run the demo:

1. Obtain a Rijksmuseum API key from the [Rijksmuseum website](https://www.rijksmuseum.nl/en/research/conduct-research/data)
2. Run the Docker container or open the `examples/rijksmuseum_demo.ipynb` notebook

