# Sandbox for executing untrusted model completions with EvalPlus.
# Built with network (to install and fetch the test suites); run with --network none and
# no host mounts -- files go in and out with `docker cp` (see run_docker.ps1).
FROM python:3.11-slim
RUN pip install --no-cache-dir "evalplus==0.3.1" \
 && useradd -m runner
USER runner
WORKDIR /home/runner
COPY --chown=runner robust_eval.py /home/runner/robust_eval.py
# Fetch both datasets, compute the reference outputs from the canonical solutions, and
# split them one file per task, all while the build still has network.
RUN python robust_eval.py shard
