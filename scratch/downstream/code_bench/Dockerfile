# Sandbox for executing untrusted model completions with EvalPlus.
# Built with network (to install and fetch the test suites); run with --network none and
# no host mounts -- files go in and out with `docker cp` (see run_docker.ps1).
FROM python:3.11-slim
RUN pip install --no-cache-dir "evalplus==0.3.1" \
 && useradd -m runner
USER runner
WORKDIR /home/runner
# Fetch both datasets into the evalplus cache now, while the build still has network.
RUN python -c "from evalplus.data import get_human_eval_plus, get_mbpp_plus; print(len(get_human_eval_plus()), len(get_mbpp_plus()))"
