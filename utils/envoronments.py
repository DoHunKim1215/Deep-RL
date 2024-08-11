def get_env_info(name: str):
    if name == 'CartPole-v1':
        return {
            'max_minutes': 20,
            'max_episodes': 10000,
            'goal_mean_100_reward': 475
        }
    elif name == 'Pendulum-v1':
        return {
            'max_minutes': 20,
            'max_episodes': 500,
            'goal_mean_100_reward': -150
        }
    elif name == 'Hopper-v5':
        return {
            'max_minutes': 300,
            'max_episodes': 10000,
            'goal_mean_100_reward': 1500
        }
    elif name == 'HalfCheetah-v5':
        return {
            'max_minutes': 300,
            'max_episodes': 10000,
            'goal_mean_100_reward': 2000
        }
    elif name == 'LunarLander-v3':
        return {
            'max_minutes': 20,
            'max_episodes': 5000,
            'goal_mean_100_reward': 250
        }
    else:
        assert False, 'No environment having such name : {}'.format(name)
