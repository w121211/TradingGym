"""
Download data
$ npx dukascopy-node -i usdjpy -from 2012-01-01 -to 2023-01-01 -t m1 -f csv --cache
"""
import pathlib

import mplfinance as mpf
import matplotlib.figure as mplfigure
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from ta import add_trend_ta, add_volatility_ta
from ta.utils import dropna
import wandb
from wandb.integration.sb3 import WandbCallback

from env.env_fx_dev import FxEntryExitSignalEnv

pathlib.Path("./download/ta").mkdir(exist_ok=True)

config = {
    "project_name": "ForexRL",
    # Data
    "train_data_raw_path": "./download/usdjpy-m1-bid-2018-01-01-2018-02-01.csv",
    "eval_data_raw_path": "./download/usdjpy-m1-bid-2018-02-01-2018-03-01.csv",
    "train_data_path": "./download/ta/usdjpy-m1-bid-2018-01-01-2018-02-01_ta.csv",
    "eval_data_path": "./download/ta/usdjpy-m1-bid-2018-02-01-2018-03-01_ta.csv",
    # Model
    "env_cls": "FxEntryExitSignalEnv",
    "model_cls": "PPO",
    "policy_net": "MlpPolicy",
    "total_timesteps": int(2e7),
    # "total_timesteps": int(1e5),
    # Env
    "asset_reversed": True,
    "obs_cols": [
        "open",
        "high",
        "low",
        "close",
        "trend_macd",
        "trend_sma_slow",
        "volatility_bbh",
        "volatility_bbl",
    ],
    "trade_commission": 0,
}
model_cls = {"PPO": PPO, "A2C": A2C}

# Init wandb
wandb.login()
run = wandb.init(
    project=config["project_name"],
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)


# Process data
def preprocess_data(df: pd.DataFrame):
    print("Preprocess data...")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = dropna(df)
    df = add_trend_ta(df, high="high", low="low", close="close")
    df = add_volatility_ta(df, high="high", low="low", close="close")
    df = df[25:][
        ["date", "timestamp"] + config["obs_cols"]
    ]  # 0 to 24 rows contains NaN

    return df


try:
    df = pd.read_csv(config["train_data_path"])
except FileNotFoundError:
    df = pd.read_csv(config["train_data_raw_path"])
    df = preprocess_data(df)
    df.to_csv(config["train_data_path"])

# Init env
env_args = {
    "asset_reversed": config["asset_reversed"],
    "obs_cols": config["obs_cols"],
    "trade_commission": config["trade_commission"],
}
env = FxEntryExitSignalEnv(df, **env_args)

# Check env, comment out to proceed
# check_env(env, warn=True)
# print("Check env success")
# raise

env = make_vec_env(lambda: env, n_envs=1)
# env = make_vec_env(FxEntryExitSignalEnv, n_envs=10, env_kwargs={"df": df})

# Train model
model = model_cls[config["model_cls"]](
    config["policy_net"],
    env,
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
).learn(
    total_timesteps=config["total_timesteps"],
    # callback=TensorboardCallback(),
    callback=WandbCallback(
        # model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

# Evaluate on different tiime period & log
try:
    df_eval = pd.read_csv(config["eval_data_path"])
except FileNotFoundError:
    df_eval = pd.read_csv(config["eval_data_raw_path"])
    df_eval = preprocess_data(df)
    df_eval.to_csv(config["eval_data_path"])

df_eval["date"] = pd.to_datetime(df_eval["date"])
df_eval.index = df_eval["date"]

# Init env
env_eval = FxEntryExitSignalEnv(df_eval, **env_args)
env_eval = make_vec_env(lambda: env_eval, n_envs=1)
reward_mean, reward_std = evaluate_policy(
    model,
    env_eval,
    n_eval_episodes=5,
    deterministic=True,
)


# Plot the trading chart of evaluation
def plot_trading(df: pd.DataFrame) -> mplfigure.Figure:
    # Run 1 episode to collect data
    action_mem: list[int] = []
    balance_mem: list[float] = []

    def on_eval_step(locals, globals):
        # print(locals)
        # print(locals["infos"][0])
        # print(locals["actions"][0])
        action_mem.append(locals["actions"][0])
        balance_mem.append(locals["infos"][0]["balance"])

    env = FxEntryExitSignalEnv(df, **env_args)
    env = make_vec_env(lambda: env, n_envs=1)
    evaluate_policy(
        model, env, n_eval_episodes=1, deterministic=False, callback=on_eval_step
    )

    # n_rows = df.shape[0]
    # mask = np.random.choice([np.NaN, 1], size=n_rows, p=[0.9, 0.1])
    # df["entry"] = df["close"] * 0.9999 * mask
    # # df['asset'] = np.random.randint(low=1000, high=10000, size=n_rows)
    # asset = [10000]
    # for _ in range(n_rows):
    #     asset.append(asset[-1] + np.random.randint(low=-10, high=10, size=1)[0])
    # df["asset"] = asset[:-1]
    # print(df)

    df = df.iloc[: len(balance_mem)]  # rows need to be equal
    addplots = [
        # mpf.make_addplot(tcdf),
        # mpf.make_addplot(low_signal,type='scatter',markersize=200,marker='^'),
        # mpf.make_addplot(high_signal,type='scatter',markersize=200,marker='v'),
        # mpf.make_addplot(
        #     df["entry"], type="scatter", markersize=5, marker="^", color="g"
        # ),
        mpf.make_addplot(balance_mem, panel=1, color="g")
        # mpf.make_addplot((df['PercentB']),panel=1,color='g')
    ]
    fig, axlist = mpf.plot(
        df,
        addplot=addplots,
        type="line",
        #  figscale=20, figsize=(160, 20)
        figsize=(15, 5),
        returnfig=True,
    )
    # mpf.plot(df, show_nontrading=True, type='line', figsize=(20, 10))

    return fig


fig = plot_trading(df_eval)
wandb.log(
    {"eval/reward_mean": reward_mean, "eval/reward_std": reward_std, "eval/plot": fig}
)

run.finish()
