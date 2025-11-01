FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENAI_API_KEY=${OPENAI_API_KEY}
EXPOSE 8501

CMD ["streamlit", "run", "app/main_interface.py"]
