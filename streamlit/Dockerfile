FROM python:3.12.4

WORKDIR /app

ADD app/ .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]