FROM nvcr.io/nvidia/pytorch:24.12-py3-igpu

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src /app/src
COPY train.py .

ENTRYPOINT ["python"]
CMD ["train.py"]
