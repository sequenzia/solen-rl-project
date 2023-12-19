import importlib
import pytz
import os
import sys

import numpy as np
import torch as th
import yaml

import gymnasium as gym

from datetime import datetime
from copy import deepcopy

from typing import Any, Dict, Optional, List, Union

from huggingface_sb3 import EnvironmentName

from rl_zoo3 import ALGOS, get_saved_hyperparams, get_wrapper_class
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path


from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize

import wandb

from dataclasses import dataclass, field


def _create_test_env(env_id: str,
                     n_envs: int = 1,
                     stats_path: Optional[str] = None,
                     seed: Optional[int] = None,
                     log_dir: Optional[str] = None,
                     should_render: bool = True,
                     hyperparams: Optional[Dict[str, Any]] = None,
                     env_kwargs: Optional[Dict[str, Any]] = None) -> VecEnv:
    """
    Create environment for testing a trained agent

    :param env_id:
    :param n_envs: number of processes
    :param stats_path: path to folder containing saved running averaged
    :param seed: Seed for random number generator
    :param log_dir: Where to log rewards
    :param should_render: For Pybullet env, display the GUI
    :param hyperparams: Additional hyperparams (ex: n_stack)
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :return:
    """

    # Create the environment and wrap it if necessary
    assert hyperparams is not None
    env_wrapper = get_wrapper_class(hyperparams)

    hyperparams = {} if hyperparams is None else hyperparams

    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    vec_env_kwargs: Dict[str, Any] = {}
    # Avoid potential shared memory issue
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

    # Fix for gym 0.26, to keep old behavior
    env_kwargs = env_kwargs or {}
    env_kwargs = deepcopy(env_kwargs)
    if "render_mode" not in env_kwargs and should_render:
        env_kwargs.update(render_mode="human")

    spec = gym.spec(env_id)

    # Define make_env here, so it works with subprocesses
    # when the registry was modified with `--gym-packages`
    # See https://github.com/HumanCompatibleAI/imitation/pull/160
    def make_env(**kwargs) -> gym.Env:
        return spec.make(**kwargs)

    env = make_vec_env(make_env,
                       n_envs=n_envs,
                       monitor_dir=log_dir,
                       seed=seed,
                       wrapper_class=env_wrapper,
                       env_kwargs=env_kwargs,
                       vec_env_cls=vec_env_cls,
                       vec_env_kwargs=vec_env_kwargs)

    if "vec_env_wrapper" in hyperparams.keys():
        vec_env_wrapper = get_wrapper_class(hyperparams, "vec_env_wrapper")
        assert vec_env_wrapper is not None
        env = vec_env_wrapper(env)
        del hyperparams["vec_env_wrapper"]

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams["normalize"]:
            print("Loading running average")
            print(f"with params: {hyperparams['normalize_kwargs']}")
            path_ = os.path.join(stats_path, "vecnormalize.pkl")
            if os.path.exists(path_):
                env = VecNormalize.load(path_, env)
                # Deactivate training and reward normalization
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {path_} not found")

        n_stack = hyperparams.get("frame_stack", 0)
        if n_stack > 0:
            print(f"Stacking {n_stack} frames")
            env = VecFrameStack(env, n_stack)
    return env


@dataclass
class Args:
    env_id: str
    algo: str
    agents_dir: str
    n_envs: int
    n_steps: int
    exp_id: int
    seed: Union[int, None]
    rewards_dir: str
    no_render: bool
    deterministic_on: bool
    stochastic_on: bool
    load_best: bool
    load_checkpoint: int
    load_last_checkpoint: bool
    norm_reward: bool
    device: str
    verbose: int
    progress: bool
    custom_objects_on: bool
    num_threads: int
    env_kwargs: Dict[str, Any]
    debug_on: bool

    run_name: str = field(init=False)
    env: EnvironmentName = field(init=False)
    ts: str = field(init=False)

    def __post_init__(self) -> None:

        ts = datetime.now(pytz.timezone('UTC')).astimezone(pytz.timezone('US/Eastern'))
        
        self.ts = ts.strftime("%Y-%m-%d-%H-%M-%S")

        self.env = EnvironmentName(self.env_id)
        self.run_name = f"{self.algo}_{self.env_id}_{self.exp_id}_{self.ts}"


@dataclass
class Episode:
     
     episode_lives: int
     episode_frame_number: int
     episode_score: int
     episode_length: int
     episode_time: float

     run_frame_number: int

@dataclass
class InferLog:

    env_idx: int

    args: Args

    n_episodes: int = 0

    info_logs: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    episode_logs: List[Episode] = field(default_factory=list, init=False)

    def add_log(self,
                env_info: Dict[str, Any]) -> Episode:

        episode_info = env_info.get("episode", {})

        if episode_info:

            episode = Episode(episode_lives=env_info["lives"],
                            episode_frame_number=env_info["episode_frame_number"],
                            episode_score=episode_info["r"],
                            episode_length=episode_info["l"],
                            episode_time=episode_info["t"],
                            run_frame_number=env_info["frame_number"])

            self.episode_logs.append(episode)
            self.info_logs.append(env_info)

            self.n_episodes += 1

            return episode

    @property
    def scores(self) -> List[float]:
        return [episode.episode_score for episode in self.episode_logs]
    
    @property
    def lengths(self) -> List[int]:
        return [episode.episode_length for episode in self.episode_logs]
    
    @property
    def times(self) -> List[float]:
        return [episode.episode_time for episode in self.episode_logs]
    

def infer(env_id: str,
          algo: str = "",
          agents_dir: str = "",
          n_envs: int = 1,
          n_steps: int = 1000,
          exp_id: int = 1,
          seed: Optional[int] = None,
          rewards_dir: str = "",
          no_render: bool = False,
          deterministic_on: bool = False,
          stochastic_on: bool = False,
          load_best: bool = True,
          load_checkpoint: str = "",
          load_last_checkpoint: bool = False,
          norm_reward: bool = False,
          device: str = "cuda",
          verbose: int = 1,
          progress: bool = False,
          custom_objects_on: bool = False,
          num_threads: int = -1,
          env_kwargs: Optional[Dict[str, Any]] = {},
          debug_on: bool = False) -> InferLog:

    args = Args(**locals())

    infer_logs: List[InferLog] = []

    wandb_run = wandb.init(project="solen-rl-project-eval",
                           name=args.run_name,
                           config={'algo': args.algo,
                                   'env_id': args.env_id,
                                   'exp_id': args.exp_id,
                                   'seed': args.seed,
                                   'n_envs': args.n_envs,
                                   'n_steps': args.n_steps,
                                   'load_best': args.load_best,
                                   'n_episodes': 0})

    for env_idx in range(args.n_envs):
        infer_logs.append(InferLog(env_idx=env_idx,
                                   args=args))

    env_name: EnvironmentName = args.env

    model_name, model_path, log_path = get_model_path(args.exp_id,
                                                      args.agents_dir,
                                                      args.algo,
                                                      env_name,
                                                      args.load_best,
                                                      args.load_checkpoint,
                                                      args.load_last_checkpoint)
    
    print(f"\nLoading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]

    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.rewards_dir if args.rewards_dir != "" else None

    env = _create_test_env(env_name.gym_id,
                           n_envs=args.n_envs,
                           stats_path=maybe_stats_path,
                           seed=args.seed,
                           log_dir=log_dir,
                           should_render=not args.no_render,
                           hyperparams=hyperparams,
                           env_kwargs=env_kwargs)
    
    kwargs = dict(seed=args.seed)
    if args.algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or args.custom_objects_on:
        custom_objects = {"learning_rate": 0.0,
                          "lr_schedule": lambda _: 0.0,
                          "clip_range": lambda _: 0.0}

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env

    model = ALGOS[args.algo].load(model_path, custom_objects=custom_objects, device=args.device, **kwargs)
    obs = env.reset()

    # Deterministic by default except for atari games
    stochastic = args.stochastic_on or (is_atari or is_minigrid) and not args.deterministic_on
    deterministic = not stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    
    successes = [] # For HER, monitor success rate
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)

    # ------------------------------------------------------------------------ #

    generator = range(args.n_steps)

    if args.progress:
        if tqdm is None:
            raise ImportError("Please install tqdm and rich to use the progress bar")
        generator = tqdm(generator)

    try:

        for gen_idx in generator:

            if args.debug_on:

                if args.debug_on and gen_idx == 0:
                    print("\n")

                if args.debug_on and not gen_idx % 1000 == 0:
                    print(f"gen_idx -> {gen_idx}")
                    print("\n")
                    print(f"info -> {infos}")

            action, lstm_states = model.predict(obs,
                                                state=lstm_states,
                                                episode_start=episode_start,
                                                deterministic=deterministic)
            
            obs, reward, done, infos = env.step(action)
            
            episode_start = done

            if not args.no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if is_atari:

                if infos:
                
                    for env_idx, env_info in enumerate(infos):

                        if env_info.get("episode"):

                            episode_log = infer_logs[env_idx].add_log(env_info=env_info)

                            print(f"\n")
                            print(f"----- ENV: {env_idx+1} EPISODE: {infer_logs[env_idx].n_episodes} -----")
                            
                            if args.debug_on:

                                print(f"\n")
                                print(f"info -> {infos}")
                                print(f"\n")

                            print(f"Episode Score: {episode_log.episode_score:.2f}")
                            print(f"Episode Length: {episode_log.episode_length}")
                            
                            wandb.log({"episode_lives": episode_log.episode_lives,
                                       "episode_score": episode_log.episode_score,
                                       "episode_length": episode_log.episode_length,
                                       "episode_time": episode_log.episode_time,
                                       "episode_frame_number": episode_log.episode_frame_number,
                                       "run_frame_number": episode_log.run_frame_number})

                            print(f"------------------------------")
                            print(f"\n")

            else:


                if args.n_envs == 1:

                    if done and not is_atari and args.verbose > 0:

                        # for env using VecNormalize, the mean reward is a normalized reward when `--norm_reward` flag is passed
                        print(f"Episode Reward: {episode_reward:.2f}")
                        print("Episode Length", ep_len)
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(ep_len)
                        episode_reward = 0.0
                        ep_len = 0

                    # Reset also when the goal is achieved when using HER
                    if done and infos[0].get("is_success") is not None:
                        if args.verbose > 1:
                            print("Success?", infos[0].get("is_success", False))

                        if infos[0].get("is_success") is not None:
                            successes.append(infos[0].get("is_success", False))
                            episode_reward, ep_len = 0.0, 0

                if args.debug_on:
                    print("\n")

    except KeyboardInterrupt:
        pass

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()

    n_episodes = sum([infer_log.n_episodes for infer_log in infer_logs])
    
    wandb.config.update({"n_episodes": n_episodes})

    wandb_run.finish()

    wandb.finish()

    return infer_logs
