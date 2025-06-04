# hash:sha256:caf6493bf0fd996952ced7e6712783664b5b5fd0ef5169205eb8548fefeb605c
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir \
    networkx==2.5.1

COPY postInstall /
RUN /postInstall
