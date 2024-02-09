import os

MAIN_DIR = "/home/ubuntu/dev/solen-rl-project"
ZOO_DIR = "/home/ubuntu/dev/repos/rl-baselines3-zoo"

os.environ["WANDB_API_KEY"] = "8c880e6018cf423b7714cf055c5fd6152e1ae117"
os.environ["WANDB_DIR"] = f"{MAIN_DIR}/logs"

LOG_DIR = f"{MAIN_DIR}/agents"
CONFIG_DIR = f"{MAIN_DIR}/configs"
TENSORBOARD_DIR = f"{MAIN_DIR}/logs/tensorboard"

PROJECT_NAME = "Solen-RL-Project-3"

SEED = 43

SAVE_FREQ = 100000
EVAL_FREQ = 10000
EVAL_EPISODES = 5

VERBOSE = 1
DEVICE = "cuda"

ROM = "Breakout"

ALGO = "ppo"

ENV_ID = f"ALE/{ROM}-v5"

TAGS = f"{ROM} {ALGO.upper()}"
CONFIG_PATH = f"{CONFIG_DIR}/{ALGO}.yml"

CMD = f"cd {MAIN_DIR} && python {ZOO_DIR}/train.py"
CMD += f" --algo {ALGO}"
CMD += f" --env {ENV_ID}"
CMD += f" --conf {CONFIG_PATH}"
CMD += f" --log-folder {LOG_DIR}"
CMD += f" --tensorboard-log {TENSORBOARD_DIR}"
CMD += f" --wandb-project-name {PROJECT_NAME}"
CMD += f" --seed {SEED}"
CMD += f" --save-freq {SAVE_FREQ}"
CMD += f" --eval-freq {EVAL_FREQ}"
CMD += f" --eval-episodes {EVAL_EPISODES}"
CMD += f" --verbose {VERBOSE}"
CMD += f" --device {DEVICE}"
CMD += f" -tags {TAGS}"
CMD += f" --track"

print(os.popen(CMD).read())
