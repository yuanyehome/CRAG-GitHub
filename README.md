# Meta CRAG KDD Cup 2024 Solution
- Team: ElectricSheep
- Team member: Ye Yuan, Jingyang Yuan, Chengwu Liu, Gongbo Sub, Siqi Li
- Mentor: Prof. Ming Zhang
- Final Score:
  - Announcement: [Winners](https://discourse.aicrowd.com/t/meta-crag-challenge-2024-winners-announcement/10786)
  - 3rd place in Task-1;
  - Prize for 5 out of 7 questions types in Task-2;

## Competition
- Homepage: [Meta CRAG KDD Cup 2024](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024)
- Baseline code: [Baseline](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/)
- See [OLD-README.md](OLD-README.md) for the old README file.

## Setup Environment
- Install the required packages by running `pip install -r requirements.txt`.
- We use conda to manage the environment. The installation procedure is shown below:
  ```bash
  conda create -n rag-vllm python=3.10
  conda activate rag-vllm
  conda install nvidia/label/cuda-11.8.0::cuda
  conda install pytorch=2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
  conda install xformers -c xformers
  export PYTHON_VERSION=310
  export VLLM_VERSION=0.4.2
  pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
  cd rag-challenge/meta-comphrehensive-rag-benchmark-starter-kit-new/
  pip install -r requirements.txt
  ```
- The installation may be different in different devices. See our `environment.yaml` for the details of the environment.

## Local Evaluation
- Run `python local_evaluation.py` for debugging the environment. This command will evaluate 10 examples in the repo.
- Run `python local_evaluation.py -h` to see all arguments.
- The Llama3-70B-4bits-GPTQ model is downloaded from `https://huggingface.co/TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ`.
  - If you don't like to download LFS files in this repo due to some speed issues, you can download the model from the link above and clone the repo with:
    ```bash
    GIT_LFS_SKIP_SMUDGE=1 git clone git@gitlab.aicrowd.com:shizueyy/crag-new.git
    ```
    Then you can use `--local-large-files-path` to specify the path of the model. Here is an example:
    ```bash
    python local_evaluation.py --use-public-test --log-dir "local_eval_logs_new/debug-task-1" --vllm_gpu_memory_utilization 0.5 --test-ratio 1.0 --local-large-files-path <LARGE-FILES-PATH>
    ```
- There are different branches for different tasks. The `main` branch is for Task-1, and the `task-{2,3}` branch is for Task-{2,3}.
