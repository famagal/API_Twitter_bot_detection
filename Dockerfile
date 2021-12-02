FROM python:3.8.12-buster
#FROM amd64/python:3.8.12-bullseye
COPY api.py /api.py
COPY setup.py /setup.py
COPY API_Twitter_bot_detection /API_Twitter_bot_detection
COPY requirements.txt /requirements.txt
COPY scripts /scripts
RUN pip install --upgrade pip
#RUN pip install -r requirements.txt
RUN pip install -e .
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
