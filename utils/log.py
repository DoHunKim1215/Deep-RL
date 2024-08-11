import logging
import time

import numpy as np
import torch
import torch.multiprocessing as mp


class Statistics:
    ERASE_LINE = '\x1b[2K'

    def __init__(self, max_episodes: int, goal_mean_100_reward: int, max_minutes: int,
                 log_period_n_secs: int, logger: logging.Logger):
        self.stats = np.empty(shape=(max_episodes, 5))
        self.stats[:] = np.nan

        self.max_episodes = max_episodes
        self.goal_mean_100_reward = goal_mean_100_reward
        self.max_minutes = max_minutes
        self.log_period_n_secs = log_period_n_secs
        self.logger = logger

        self.episode_reward = []
        self.episode_timestep = []
        self.episode_exploration = []
        self.evaluation_scores = []
        self.episode_seconds = []

        self.training_time = 0.
        self.episode_start = 0.
        self.training_start = 0.
        self.last_debug_time = float('-inf')

    def start_training(self):
        self.training_start = time.time()

    def prepare_before_episode(self):
        self.episode_reward.append(0.0)
        self.episode_timestep.append(0.0)
        self.episode_exploration.append(0.0)
        self.episode_start = time.time()

    def add_one_step_data(self, reward: float, is_exploration: bool):
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += float(is_exploration)

    def calculate_elapsed_time(self):
        episode_elapsed = time.time() - self.episode_start
        self.episode_seconds.append(episode_elapsed)
        self.training_time += episode_elapsed

    def append_evaluation_score(self, evaluation_score: float):
        self.evaluation_scores.append(evaluation_score)

    def process_after_episode(self, episode_idx: int):
        # calculate stats
        cumulated_step = int(np.sum(self.episode_timestep))
        mean_10_reward = np.mean(self.episode_reward[-10:])
        std_10_reward = np.std(self.episode_reward[-10:])
        mean_100_reward = np.mean(self.episode_reward[-100:])
        std_100_reward = np.std(self.episode_reward[-100:])
        mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
        std_100_eval_score = np.std(self.evaluation_scores[-100:])
        lst_100_exp_rat = np.array(self.episode_exploration[-100:]) / np.array(self.episode_timestep[-100:])
        mean_100_exp_rat = np.mean(lst_100_exp_rat)
        std_100_exp_rat = np.std(lst_100_exp_rat)

        wallclock_elapsed = time.time() - self.training_start
        self.stats[episode_idx] = (cumulated_step,
                                   mean_100_reward,
                                   mean_100_eval_score,
                                   self.training_time,
                                   wallclock_elapsed)

        # logging condition
        reached_debug_time = time.time() - self.last_debug_time >= self.log_period_n_secs

        # termination conditions
        reached_max_minutes = wallclock_elapsed >= self.max_minutes * 60
        reached_max_episodes = episode_idx + 1 >= self.max_episodes
        reached_goal_mean_reward = mean_100_eval_score >= self.goal_mean_100_reward
        training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward

        # logging
        if reached_debug_time or training_is_over:
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.training_start))
            debug_message = '[{}] EPISODE {:04}, STEP {:06}, '
            debug_message += 'REWARD[-10:] {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'REWARD[-100:] {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'EXP_RATE[-100:] {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'EVAL_SCORE[-100:] {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode_idx, cumulated_step, mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score
            )
            self.logger.info(debug_message)
            self.last_debug_time = time.time()

        if training_is_over:
            self.logger.info('--> reached max_minutes {} max_episodes {} goal_mean_reward {}'
                             .format('(o)' if reached_max_minutes else '(x)',
                                     '(o)' if reached_max_episodes else '(x)',
                                     '(o)' if reached_goal_mean_reward else '(x)'))

        return training_is_over

    def get_total(self):
        wallclock_time = time.time() - self.training_start
        return self.stats, self.training_time, wallclock_time

    def get_total_steps(self):
        return np.sum(self.episode_timestep)


class MultiEnvStatistics:
    ERASE_LINE = '\x1b[2K'

    def __init__(self, n_workers: int, max_episodes: int, goal_mean_100_reward: int,
                 max_minutes: int, log_period_n_secs: int, logger: logging.Logger):
        self.n_workers = n_workers

        self.stats = np.empty(shape=(max_episodes, 5))
        self.stats[:] = np.nan

        self.max_episodes = max_episodes
        self.goal_mean_100_reward = goal_mean_100_reward
        self.max_minutes = max_minutes
        self.log_period_n_secs = log_period_n_secs
        self.logger = logger

        self.episode_reward = []
        self.episode_timestep = []
        self.episode_exploration = []
        self.evaluation_scores = []
        self.episode_seconds = []

        self.running_timestep = np.array([[0.], ] * self.n_workers)
        self.running_reward = np.array([[0.], ] * self.n_workers)
        self.running_exploration = np.array([[0.], ] * self.n_workers)
        self.running_seconds = np.array([[time.time()], ] * self.n_workers)

        self.episode = 0
        self.training_time = 0.
        self.episode_start = 0.
        self.training_start = 0.
        self.last_debug_time = float('-inf')

    def start_training(self):
        self.training_start = time.time()

    def add_one_step_data(self, rewards: np.ndarray, is_exploration: np.ndarray):
        self.running_reward += rewards
        self.running_timestep += 1
        self.running_exploration += is_exploration[:, np.newaxis].astype(np.int32)

    def reset_worker_finishing_episode(self, episode_idx: int, rank: int, eval_score: float, episode_done_time: float):
        self.episode_timestep.append(self.running_timestep[rank][0])
        self.episode_reward.append(self.running_reward[rank][0])
        self.episode_exploration.append(self.running_exploration[rank][0] / self.running_timestep[rank][0])
        self.episode_seconds.append(episode_done_time - self.running_seconds[rank][0])
        self.training_time += self.episode_seconds[-1]
        self.evaluation_scores.append(eval_score)

        mean_100_reward = np.mean(self.episode_reward[-100:])
        mean_100_eval_score = np.mean(self.evaluation_scores[-100:])

        total_step = int(np.sum(self.episode_timestep))
        wallclock_elapsed = time.time() - self.training_start
        self.stats[episode_idx] = total_step, mean_100_reward, mean_100_eval_score, self.training_time, wallclock_elapsed

    def process_after_episode_done(self, episode_idx: int, terminated: np.ndarray) -> bool:
        # reset running variables for next time around
        self.running_timestep *= 1 - terminated
        self.running_reward *= 1 - terminated
        self.running_exploration *= 1 - terminated
        self.running_seconds[terminated.astype(np.bool_)] = time.time()

        mean_10_reward = np.mean(self.episode_reward[-10:])
        std_10_reward = np.std(self.episode_reward[-10:])
        mean_100_reward = np.mean(self.episode_reward[-100:])
        std_100_reward = np.std(self.episode_reward[-100:])
        mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
        std_100_eval_score = np.std(self.evaluation_scores[-100:])
        mean_100_exp_rat = np.mean(self.episode_exploration[-100:])
        std_100_exp_rat = np.std(self.episode_exploration[-100:])

        # debug stuff
        reached_debug_time = time.time() - self.last_debug_time >= self.log_period_n_secs
        reached_max_minutes = time.time() - self.training_start >= self.max_minutes * 60
        reached_max_episodes = episode_idx + self.n_workers >= self.max_episodes
        reached_goal_mean_reward = mean_100_eval_score >= self.goal_mean_100_reward
        training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward

        if reached_debug_time or training_is_over:
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.training_start))
            debug_message = '[{}] EP {:04}, STEP {:06}, '
            debug_message += 'REWARD[-10:] {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'REWARD[-100:] {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'EXP_RATE[-100:] {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'EVAL[-100:] {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode_idx - 1, int(np.sum(self.episode_timestep)), mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat, mean_100_eval_score,
                std_100_eval_score)
            self.logger.info(debug_message)
            self.last_debug_time = time.time()

        if training_is_over:
            self.logger.info('--> reached max_minutes {} max_episodes {} goal_mean_reward {}'
                             .format('(o)' if reached_max_minutes else '(x)',
                                     '(o)' if reached_max_episodes else '(x)',
                                     '(o)' if reached_goal_mean_reward else '(x)'))

        return training_is_over

    def get_result(self):
        wallclock_time = time.time() - self.training_start
        return self.stats, self.training_time, wallclock_time

    def get_total_steps(self):
        return np.sum(self.episode_timestep)


class MultiLearnerStatistics:
    ERASE_LINE = '\x1b[2K'

    def __init__(self, n_workers: int, get_out_lock: mp.Lock,
                 max_episodes: int, goal_mean_100_reward: int, max_minutes: int, log_period_n_secs: int):
        self.n_workers = n_workers
        self.get_out_lock = get_out_lock

        self.stats = torch.zeros([max_episodes, 5]).share_memory_()

        self.max_episodes = max_episodes
        self.goal_mean_100_reward = goal_mean_100_reward
        self.max_minutes = max_minutes
        self.log_period_n_secs = log_period_n_secs

        self.episode = torch.zeros(1, dtype=torch.int).share_memory_()
        self.current_episode = torch.zeros(n_workers, dtype=torch.int).share_memory_()

        self.episode_reward = torch.zeros([max_episodes]).share_memory_()
        self.episode_timestep = torch.zeros([max_episodes], dtype=torch.int).share_memory_()
        self.episode_exploration = torch.zeros([max_episodes]).share_memory_()
        self.evaluation_scores = torch.zeros([max_episodes]).share_memory_()
        self.episode_seconds = torch.zeros([max_episodes]).share_memory_()

        self.total_episode_rewards = torch.zeros(n_workers, dtype=torch.float).share_memory_()
        self.total_episode_steps = torch.zeros(n_workers, dtype=torch.int).share_memory_()
        self.total_episode_exploration = torch.zeros(n_workers, dtype=torch.float).share_memory_()

        self.episode_start = torch.zeros(n_workers, dtype=torch.float64).share_memory_()
        self.training_start = 0.
        self.last_debug_time = torch.zeros(1, dtype=torch.float64).share_memory_()

        self.reached_max_minutes = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_max_episodes = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_goal_mean_reward = torch.zeros(1, dtype=torch.int).share_memory_()

    def start_training(self):
        self.training_start = time.time()

    def prepare_before_work(self, rank: int):
        if rank == 0:
            self.last_debug_time = float('-inf')
        self.current_episode[rank] = self.episode.add_(1).item() - 1

    def get_current_episode(self, rank: int):
        return self.current_episode[rank]

    def prepare_before_episode(self, rank: int):
        assert 0 <= rank < self.n_workers
        self.episode_start[rank] = time.time()
        self.total_episode_rewards[rank] = 0.
        self.total_episode_steps[rank] = 0
        self.total_episode_exploration[rank] = 0.

    def add_one_step_data(self, rank: int, reward: float, is_exploration: bool):
        assert 0 <= rank < self.n_workers
        self.total_episode_rewards[rank].add_(reward)
        self.total_episode_steps[rank].add_(1)
        self.total_episode_exploration[rank].add_(float(is_exploration))

    def calculate_elapsed_time(self, rank: int):
        assert 0 <= rank < self.n_workers
        episode_elapsed = time.time() - self.episode_start[rank].item()
        self.episode_seconds[self.current_episode[rank]].add_(episode_elapsed)

    def append_evaluation_score(self, rank: int, evaluation_score: float):
        assert 0 <= rank < self.n_workers
        self.evaluation_scores[self.current_episode[rank]].add_(evaluation_score)

    def process_after_episode(self, rank: int):
        assert 0 <= rank < self.n_workers
        current_rank_episode = self.current_episode[rank]

        self.episode_timestep[current_rank_episode].add_(self.total_episode_steps[rank])
        self.episode_reward[current_rank_episode].add_(self.total_episode_rewards[rank])
        self.episode_exploration[current_rank_episode].add_(
            self.total_episode_exploration[rank] / self.total_episode_steps[rank]
        )

        # calculate stats
        cumulated_step = self.episode_timestep[:current_rank_episode + 1].sum().item()

        mean_10_reward = self.episode_reward[:current_rank_episode + 1][-10:].mean().item()
        mean_100_reward = self.episode_reward[:current_rank_episode + 1][-100:].mean().item()
        mean_100_eval_score = self.evaluation_scores[:current_rank_episode + 1][-100:].mean().item()
        mean_100_exp_rat = self.episode_exploration[:current_rank_episode + 1][-100:].mean().item()

        if current_rank_episode != 0:
            std_10_reward = self.episode_reward[:current_rank_episode + 1][-10:].std().item()
            std_100_reward = self.episode_reward[:current_rank_episode + 1][-100:].std().item()
            std_100_eval_score = self.evaluation_scores[:current_rank_episode + 1][-100:].std().item()
            std_100_exp_rat = self.episode_exploration[:current_rank_episode + 1][-100:].std().item()
        else:
            std_10_reward = 0.
            std_100_reward = 0.
            std_100_eval_score = 0.
            std_100_exp_rat = 0.

        training_time = self.episode_seconds[:current_rank_episode + 1].sum().item()
        wallclock_elapsed = time.time() - self.training_start

        self.stats[current_rank_episode][0].add_(cumulated_step)
        self.stats[current_rank_episode][1].add_(mean_100_reward)
        self.stats[current_rank_episode][2].add_(mean_100_eval_score)
        self.stats[current_rank_episode][3].add_(training_time)
        self.stats[current_rank_episode][4].add_(wallclock_elapsed)

        with self.get_out_lock:
            potential_next_episode = self.episode.item()
            self.reached_goal_mean_reward.add_(mean_100_eval_score >= self.goal_mean_100_reward)
            self.reached_max_minutes.add_(time.time() - self.training_start >= self.max_minutes * 60)
            self.reached_max_episodes.add_(potential_next_episode >= self.max_episodes)
            training_is_over = self.reached_max_episodes or self.reached_max_minutes or self.reached_goal_mean_reward

        # logging
        if rank == 0 and (time.time() - self.last_debug_time >= self.log_period_n_secs or training_is_over):
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.training_start))
            debug_message = '[{}] EPISODE {:04}, STEP {:06}, '
            debug_message += 'REWARD[-10:] {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'REWARD[-100:] {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'EXP_RATE[-100:] {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'EVAL_SCORE[-100:] {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, current_rank_episode + 1, cumulated_step, mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score
            )
            print(debug_message, flush=True)
            self.last_debug_time = time.time()

        return training_is_over

    def log_finishing_work(self, rank: int):
        if rank == 0:
            print('--> reached max_minutes {} max_episodes {} goal_mean_reward {}'
                  .format('(o)' if self.reached_max_minutes else '(x)',
                          '(o)' if self.reached_max_episodes else '(x)',
                          '(o)' if self.reached_goal_mean_reward else '(x)'), flush=True)

    def get_total(self):
        wallclock_time = time.time() - self.training_start
        final_episode = self.episode.item()
        training_time = self.episode_seconds[:final_episode + 1].sum().item()
        stats = self.stats.numpy()
        stats[final_episode:, ...] = np.nan
        return stats, training_time, wallclock_time

    def go_to_next_episode(self, rank: int):
        self.current_episode[rank] = self.episode.add_(1).item() - 1
