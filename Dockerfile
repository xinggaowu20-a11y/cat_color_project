FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
ENV PORT=7860

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
