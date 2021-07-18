FROM tiangolo/uwsgi-nginx-flask:python3.6
WORKDIR /ml_prod/
ADD . /ml_prod
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD python3 pipeline.py