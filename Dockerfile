FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /opt/tfc
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu"
COPY requirements-docker.txt ./
RUN python -m pip install --no-cache-dir -r requirements-docker.txt
COPY code/TFC/model.py ./code/TFC/model.py
COPY code/config_files/HAR_Configs.py ./code/config_files/HAR_Configs.py
COPY docker_app/ ./docker_app/
COPY models/tfc_har_gesture_model.pt ./models/tfc_har_gesture_model.pt
COPY models/tfc_har_gesture_classifier.pt ./models/tfc_har_gesture_classifier.pt
RUN python -m docker_app.verify_model
RUN useradd --create-home --uid 10001 tfcuser && chown -R tfcuser:tfcuser /opt/tfc
USER tfcuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 CMD python -c "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:8000/health',timeout=4).status==200"
CMD ["python", "-m", "uvicorn", "docker_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
