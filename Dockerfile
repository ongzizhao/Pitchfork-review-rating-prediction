FROM python:3.7
COPY . /app
EXPOSE 8080
WORKDIR /app
RUN pip install -r requirements.txt
CMD python app.py