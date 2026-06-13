FROM python:3.12

RUN pip install --no-cache-dir \
    mlflow \
    pymysql \
    boto3 && \
    mkdir -p /mlflow

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root ${BUCKET} \
    --backend-store-uri mysql+pymysql://${USERNAME}:${PASSWORD}@${HOST}:${PORT}/${DATABASE}
