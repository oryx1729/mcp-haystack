FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    curl \
    git \
    gnupg \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install Jupyter and haystack-ai
RUN pip install --no-cache-dir \
    jupyter \
    notebook \
    haystack-ai>=2.9.0 \
    nest-asyncio

# Copy the local MCP integration and install it
COPY . /mcp-haystack
RUN pip install -e /mcp-haystack

# Clone and install the rijksmuseum-mcp with specific commit
RUN git clone https://github.com/r-huijts/rijksmuseum-mcp.git /rijksmuseum-mcp \
    && cd /rijksmuseum-mcp \
    && git checkout af9c2a2dba1e709f2193a59bb3fee3a3b66380b5 \
    && npm install \
    && npm run build

# Create a directory for notebooks and copy the example notebook
RUN mkdir -p /app/notebooks
COPY examples/rijksmuseum_demo.ipynb /app/notebooks/

# Create a directory for user data
RUN mkdir -p /app/user_data

# Expose Jupyter port
EXPOSE 8888

# Set environment variables
ENV NODE_PATH=/usr/bin/node
ENV MCP_SERVER_PATH=/rijksmuseum-mcp/build/index.js

# Set the working directory to where the notebooks are
WORKDIR /app/notebooks

# Set the entrypoint to Jupyter Notebook with the demo notebook
ENTRYPOINT ["jupyter", "notebook", "rijksmuseum_demo.ipynb", "--ip=0.0.0.0", "--port=8888", "--allow-root"] 