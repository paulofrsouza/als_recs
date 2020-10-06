FROM python:3.6-slim

RUN apt update && apt install -y build-essential

WORKDIR /usr/src/app

RUN mkdir als_recs
COPY als_recs ./als_recs 

COPY requirements.txt  setup.py ./
RUN pip3 install -r requirements.txt
RUN pip3 install -e .

CMD ["python3", "./als_recs/als_cli.py"]