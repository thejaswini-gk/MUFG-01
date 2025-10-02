# Use Python official image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY app.py .
COPY heart_disease_model.pkl .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
