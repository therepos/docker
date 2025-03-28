FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies including Tesseract and poppler-utils
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    calibre \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm get-pip.py

# Copy only requirements first (better Docker layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

# Expose the API port
EXPOSE 8000

# Set the command to run the app using Uvicorn
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]    # Uncomment for testing
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


