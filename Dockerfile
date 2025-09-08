FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set build arguments
# Run "id fantoni" in the bash terminal and set UID and GID of fantoni (1003 in this case).
ARG UNAME=pytorch
ARG UID=1003
ARG GID=1003 

# Create the user inside the container
RUN groupadd -g $GID -o $UNAME \
    && useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

# Create app directory and set permissions
RUN mkdir /app \
    && chown -R $UID:$GID /app 

# Switch to the new user
USER $UNAME
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src /app/src
COPY train.py .
COPY config.yaml .

# Uncomment to run python script
#ENTRYPOINT ["python"]
#CMD ["train.py"]

# Uncomment to run jupiter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]