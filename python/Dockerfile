# Dockerfile for tangent_py development and testing
# Uses Mambaforge for fast cppyy installation via pre-built binaries

FROM condaforge/mambaforge:latest

# Install cppyy with Python 3.10 (cppyy not available for 3.12 yet)
RUN mamba install -y -c conda-forge python=3.10 cppyy numpy pytest && \
    mamba clean -afy

# Set working directory
WORKDIR /tangent

# Copy the project
COPY . .

# Install tangent_py in editable mode
RUN pip install -e python/

# Default command: run tests
CMD ["pytest", "python/tests/", "-v"]
