# RepCounter container for the NVIDIA Jetson Orin Nano (JetPack 6).
#
# Build:  docker build -t repcounter:dev .
# Run:    docker run -it --rm --runtime nvidia -p 8501:8501 repcounter:dev
ARG BASE_IMAGE=ultralytics/ultralytics:latest-jetson-jetpack6
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit frontend port (used later in the project)
EXPOSE 8501

# Default command: count the reps in a sample push-up clip
CMD ["python3", "backend/rep_counter.py", "data/clips/pushup1.mp4", "pushup"]
