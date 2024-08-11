import numpy as np
import torch.multiprocessing as mp


class MultiprocessEnv(object):

    def __init__(self, make_env_fn, make_env_kwargs, seed, n_workers):
        self.make_env_fn = make_env_fn
        self.make_env_kwargs = make_env_kwargs
        self.seed = seed
        self.n_workers = n_workers

        self.pipes = []
        self.workers = []
        for rank in range(n_workers):
            parent_end, child_end = mp.Pipe()
            self.pipes.append(parent_end)
            env = self.make_env_fn(**self.make_env_kwargs)
            env.reset(seed=self.seed + rank)
            self.workers.append(mp.Process(target=self.work, args=(env, child_end)))
        for w in self.workers:
            w.start()

    def reset(self, rank: int, **kwargs):
        self.send_msg(('reset', kwargs), rank)
        o, _ = self.pipes[rank].recv()
        return o

    def reset_all(self, **kwargs):
        self.broadcast_msg(('reset', kwargs))
        return np.vstack([parent_end.recv()[0] for parent_end in self.pipes])

    def step(self, actions):
        assert len(actions) == self.n_workers
        for rank in range(self.n_workers):
            self.send_msg(('step', {'action': actions[rank]}), rank)

        results = []
        for rank in range(self.n_workers):
            observation, reward, terminated, truncated, _ = self.pipes[rank].recv()
            results.append((observation,
                            np.array(reward, dtype=np.float32),
                            np.array(terminated or truncated, dtype=np.float32),
                            truncated))
        return [np.vstack(block) for block in np.array(results, dtype=object).T]

    @staticmethod
    def work(env, worker_end):
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == 'reset':
                worker_end.send(env.reset(**kwargs))
            elif cmd == 'step':
                worker_end.send(env.step(**kwargs))
            elif cmd == 'close':
                # including close command
                env.close(**kwargs)
                del env
                worker_end.close()
                break
            else:
                assert False, 'Wrong command {}'.format(cmd)

    def close(self, **kwargs):
        self.broadcast_msg(('close', kwargs))
        for w in self.workers:
            w.join()

    def send_msg(self, msg, rank):
        self.pipes[rank].send(msg)

    def broadcast_msg(self, msg):
        for parent_end in self.pipes:
            parent_end.send(msg)
