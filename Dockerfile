FROM python:3.7
COPY . /
EXPOSE 5000
WORKDIR WORKDIR /
RUN pip install -r requirements.txt
CMD python app.py