FROM nvcr.io/nvidia/pytorch:24.08-py3

ARG UNAME=pytorch
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME \
    && useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

RUN mkdir /app \
    && chown -R $UID:$GID /app 

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src /app/src
COPY train.py .
COPY config.yaml .

# Uncomment to run python script
ENTRYPOINT ["python"]
CMD ["train.py"]

# Uncomment to run jupiter notebook
#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
