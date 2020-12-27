FROM continuumio/anaconda3:4.4.0
EXPOSE 5000
WORKDIR WORKDIR /app
RUN pip install -r requirements.txt
CMD python app.py