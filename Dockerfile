FROM nvcr.io/nvidia/pytorch:24.08-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src /app/src
COPY train.py .
COPY config.yaml .

ENTRYPOINT ["python"]
CMD ["train.py"]
