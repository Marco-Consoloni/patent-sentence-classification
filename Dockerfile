FROM nvcr.io/nvidia/pytorch:24.08-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src /app/src
COPY train.py .
COPY config.yaml .
COPY test.py .

# Uncomment to run python script
#ENTRYPOINT ["python"]
#CMD ["train.py"]

# Uncomment to run jupiter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
