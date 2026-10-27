FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py steel_model.py prepare_data.py ./
COPY demo ./demo
ENTRYPOINT ["python", "main.py"]
CMD ["train", "--demo"]
