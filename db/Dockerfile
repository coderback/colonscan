# db/Dockerfile
FROM postgres:latest

USER root
RUN apt-get update \
 && apt-get install -y procps \
 && rm -rf /var/lib/apt/lists/*

USER postgres
