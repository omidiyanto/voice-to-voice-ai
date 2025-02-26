FROM python:3.9-slim-bullseye
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
EXPOSE 7860
CMD ["python","app.py"]