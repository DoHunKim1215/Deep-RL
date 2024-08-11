from utils.plot import plt_many_agents

if __name__ == '__main__':
    paths = {
        'DQN': 'results/DQN/logs/env_CartPole-v1_model_DQN_date_2024-07-31-19-57-06.npy',
        'DDQN': 'results/DDQN/logs/env_CartPole-v1_model_DDQN_date_2024-07-24-16-04-07.npy',
        'Dueling DDQN': 'results/DuelingDDQN/logs/env_CartPole-v1_model_DuelingDDQN_date_2024-07-24-20-41-55.npy',
        'Dueling DDQN + PER': 'results/DuelingDDQN+PER/logs/env_CartPole-v1_model_DuelingDDQN+PER_date_2024-08-08-17-09-07.npy',
    }

    plt_many_agents(paths, 'plot_cartpole_value_without_nfq.png')
