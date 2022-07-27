FROM continuumio/miniconda3

WORKDIR /work

# create conda env
COPY ai4eo_hyper.yml .
RUN conda env create -f ai4eo_hyper.yml

# make RUN commands use new env
SHELL ["conda", "run", "-n", "ai4eo_hyper", "/bin/bash", "-c"]

# copy the code
COPY ["rf_train.py", "jobs/submit_trial.sh", "./"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ai4eo_hyper", "squeue"]
