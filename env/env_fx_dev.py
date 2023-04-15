import time
import random
from copy import deepcopy
from typing import Any
from datetime import datetime

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd
from attrs import define
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


@define
class LimitOrder:
    """Non-vectorized limit order"""

    # price: npt.NDArray[np.float32]
    # share: npt.NDArray[np.float32]
    price: float
    share: float
    issued_datetime: datetime  # Used to calculate elapsed time
    start_price: float  # Used to calculate profit/loss


class FxEntryExitSignalEnv(gym.Env):
    """
    State
        Step t
        Asset price at t
        Portfolio at t
            Assets holdings (in shares)
            Balance (in USD): Sum of current assets value
    Action
        Discrete action for LONG, SHORT, NOTHING
            It removes the amount of trade and only focus on the trend of stock to take profit.
            Open a position to long/short an asset, and close by the agent or automatically triggered the stop-loss/profit-taker.
            If close earlier than SL/PT, what happened?
        Number of shares to buy/sell. Continuous action for (a1_shares, a2_shares, a_n_shares....), range from (-1, 1)
    Reward
        r_t = ...
            Encourage: holding position longer, stoploss
            Discourage: low profit, no trade, reverse position (eg BUY 10 and then SELL 20)
    """

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        df: pd.DataFrame,  # [time, open, high, low, close, adjusted_close, volume*, indicator_1, ...] * n_assets * num_t_steps. Index from 0 to t-1 of t steps
        obs_cols=[
            "open",
            "high",
            "low",
            "close",
        ],  # obsavable info at step t, assume all can be convert to float
        obs_backward_window=32,
        asset_reversed=False,  # reversed: usdjpy, not-reversed: eurusd
        trade_n_assets=1,  # exclude the primary asset, ie USD
        trade_commission=0.001,  # in percentage
        agent_start_usd=1e4,  # agent's starting cash in USD
        episode_random_start=True,  # choose a random step as the starting point
        episode_steps=1000,  # terminate episode if exceeded
    ) -> None:
        super(FxEntryExitSignalEnv, self).__init__()
        self.df = df
        self.obs_cols = obs_cols
        self.obs_backward_window = obs_backward_window
        self.asset_reversed = asset_reversed
        self.trade_n_assets = trade_n_assets
        self.trade_commission = trade_commission
        self.agent_start_usd = agent_start_usd
        self.agent_max_loss = agent_start_usd  # terminate if exceeded
        self.agent_entry_scalar = int(
            (
                agent_start_usd
                if asset_reversed
                else agent_start_usd / self.df["close"][0]
            )
            / 10
        )  # equivalent amount of initial USD, and split into 10 packs
        self.episode_random_start = episode_random_start
        self.episode_steps = episode_steps

        # print("agent_entry_scalar", self.agent_entry_scalar)

        self.t = obs_backward_window  # starter step
        self.t_elapsed = 0  # elapsed t of current episode
        self.limit_orders_open_t: list[LimitOrder] = []  # current open limit orders
        self.limit_orders_done: list[LimitOrder] = []  # completed limit orders

        self.action_space = spaces.Discrete(3)  # 3 actions: long, short, nothing
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            # shape=((1 + self.n_assets) + len(df_obs_cols) * self.n_assets,),
            shape=(self.obs_backward_window, len(self.obs_cols)),
            dtype=np.float32,
        )  # [holding_usd, holding_eur, ..., holding_asset_n, price_eur, ..., price_asset_n]

        # Memories, record at every step
        self.agent_holdings_memo: list[
            npt.NDArray[np.float32]
        ] = []  # Agent's holdings, eg [USD_amount, EUR_amout, ...]
        self.agent_balance_memo: list[float] = []  # In USD

    def reset(self) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        self.t = (
            np.random.randint(
                low=self.obs_backward_window, high=len(self.df) - self.episode_steps
            )
            if self.episode_random_start
            else self.obs_backward_window
        )
        self.t_elapsed = 0
        self.limit_orders_open_t = []
        self.limit_orders_done = []

        # TODO: Multi asset
        holdings_t = np.append(
            [self.agent_start_usd], np.zeros(self.trade_n_assets)
        ).astype(
            np.float32
        )  # agent's initial holdings, in the form of [USD amount, EUR amout, ...]
        obs_t = self.df[self.t - self.obs_backward_window : self.t][
            self.obs_cols
        ].to_numpy(dtype=np.float32)
        # state_0 = np.concatenate([holdings_0, obs_0], dtype=np.float32)
        balance_t = np.dot(
            [1.0, self.df["close"][self.t]], holdings_t
        )  # use close price to calculate the final balance at step t

        # Logging
        self.agent_holdings_memo = [holdings_t]
        self.agent_balance_memo = [balance_t]
        info = {
            "holdings": self.agent_holdings_memo[-1],
            "balance": self.agent_balance_memo[-1],
        }

        return obs_t, info

    def step(
        self, action: npt.NDArray[np.float16]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        # Action 0: buy, 1: sell, 2: nothing
        if action == 0:
            buysell = 1.0
        elif action == 1:
            buysell = -1.0
        elif action == 2:
            buysell = 0.0
        else:
            print("Unexpected action", action)
            raise

        # buysell = np.where(action > 0.7, 1, action)
        # buysell = np.where(action < -0.7, -1, buysell)
        # buysell = np.where(-0.7 < action < 0.7, -1, buysell)
        buysell_shares = (
            buysell * self.agent_entry_scalar
        )  # Scale action by the scalar.

        # stoploss = np.array({"qty": -buysell, "price": price_cur * 0.9})
        # profittaker = np.array({"qty": -buysell, "price": price_cur * 1.1})
        # self.limit_orders.append(stop)

        # Process previous limit orders (include stop loss, profit taker)
        # Process incoming orders, either immediate execute or push to the stack
        # for e in orders:
        #     self._order_execute(e)

        # Process market order
        holdings_after, balance_after = self._transact(
            self.agent_holdings_memo[-1],
            self.df["open"][
                self.t
            ],  # open price is the nearest price after sending the market order
            buysell_shares,
        )

        # Reward
        balance_diff = balance_after - self.agent_balance_memo[-1]
        reward = balance_diff

        # Next step
        self.t += 1
        self.t_elapsed += 1
        self.agent_balance_memo.append(balance_after)
        self.agent_holdings_memo.append(holdings_after)
        info_t = {
            "holdings": self.agent_holdings_memo[-1],
            "balance": self.agent_balance_memo[-1],
        }

        # Truncate if reached the end of available data

        if self.t_elapsed == self.episode_steps or self.t == len(self.df):
            truncated = True
            obs_t = np.zeros(shape=(self.obs_backward_window, len(self.obs_cols)))
            reward = 0.0
            return obs_t, reward, False, truncated, info_t
        else:
            obs_t = self.df[self.t - self.obs_backward_window : self.t][
                self.obs_cols
            ].to_numpy(dtype=np.float32)

        # Terminate if exceeded the maximum loss
        balance_diff_from_start = balance_after - self.agent_balance_memo[0]
        if balance_diff_from_start < -self.agent_max_loss:
            terminated = True
            return obs_t, reward, terminated, False, info_t

        # print(action, reward)

        return obs_t, reward, False, False, info_t

    def _transact(
        self,
        holdings: npt.NDArray[np.float32],
        price: float,
        buysell_shares: float,  # 1-D array (n_assets, )
    ):
        """Process the transaction and update the given porfolio."""

        if self.asset_reversed:
            holdings_after = (
                holdings
                + np.array([buysell_shares, -price * buysell_shares])
                - np.array([0, self.trade_commission * price * abs(buysell_shares)])
            )
            balance_after = np.dot([1, 1 / price], holdings)  # in usd
        else:
            holdings_after = (
                holdings
                + np.array([-price * buysell_shares, buysell_shares])
                - np.array([self.trade_commission * price * abs(buysell_shares), 0])
            )
            balance_after = np.dot([1, price], holdings)

        return holdings_after, balance_after

    # def _order_execute(self, order: MarketOrder):
    #     """Process orders
    #     Market order (lazy)
    #         Execute in the current step.
    #         Fulfilled price = the current asset close price.
    #         Fulfilled quantiy = setter quantity *Not realistic, need to pay attention to the asset's liqduitiy.
    #     Limit order (lazy):
    #     - Push the new orders to the stack
    #     - For each order in the stack, try to execute
    #     Limit order (serious):
    #     - Execute the stack first.
    #     - Push incoming orders to the stack (and not executed). Because in realilty the order can only be effective after the current tick.
    #     Stop loss, Profit taken:
    #     -
    #     """
    #     pass

    # def _execute_limit_orders(self) -> None:
    #     """
    #     Lazy: Issued at step-t is executed at step-t
    #     Serious: Issued at step-t is executed after step-t
    #     """
    #     new_stack: list[LimitOrder] = []
    #     for e in self.limit_orders_open_t:
    #         # If limit price is inside price_candle_t, a trade happened
    #         if price_candle_t.min < e.price < price_candle_t.max:
    #             self.transact(e.price, e.buysell_shares)
    #         else:
    #             new_stack.append(e)
    #     self.cur_limit_orders = new_stack
    #     # if (order.limit_price * order.buysell - order.buysell * price_t) >= 0:
    #     #     self.transact()
