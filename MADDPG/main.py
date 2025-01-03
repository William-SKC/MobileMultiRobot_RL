import argparse
import os
import wandb

import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3, simple_speaker_listener_v4, simple_push_v3, simple_v3, simple_nav_v1

from MADDPG import MADDPG

vis_setting = False

def get_env(env_name, vis_setting = False, ep_len=25):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(render_mode="rgb_array", max_cycles=ep_len)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=ep_len)
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode="rgb_array", max_cycles=ep_len)
    if env_name == 'simple_speaker_listener_v4':
        new_env = simple_speaker_listener_v4.parallel_env(render_mode="rgb_array", max_cycles=ep_len)
    if env_name == 'simple_push_v3':
        new_env = simple_push_v3.parallel_env(render_mode="rgb_array", max_cycles=ep_len)
    if env_name == 'simple_v3':
        new_env = simple_v3.parallel_env(render_mode="rgb_array", max_cycles=ep_len)
    if env_name == 'simple_nav_v1':
        new_env = simple_nav_v1.parallel_env(render_mode="rgb_array", max_cycles=ep_len, vis = vis_setting, force_based = False)



    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='simple_nav_v1', help='name of the env',
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3', 'simple_speaker_listener_v4', 'simple_push_v3', 'simple_v3' 'simple_nav_v1'])
    parser.add_argument('--episode_num', type=int, default=50000,
                        help='total episode num during training procedure') #30000
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')#25
    parser.add_argument('--learn_interval', type=int, default=100,
                        help='steps interval between learning time')#100
    parser.add_argument('--random_steps', type=int, default=5e3,
                        help='random steps before the agent start to learn')#5e4
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e3), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=128, help='batch-size of replay buffer') #1024
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    args = parser.parse_args()

    # create folder to save result
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    #create environment for experiment
    env, dim_info = get_env(args.env_name, vis_setting, args.episode_length)
    print("dim_info: ", dim_info)
    
    # model
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                    result_dir, vis = vis_setting)

    """Start Training"""
    experiment_name = f"MADDPG_{args.env_name}_batch_size_{args.batch_size}"
    wandb.init(project="mbrl-nfo", group="Delta_Baseline2", name=experiment_name)
    wandb.config.update(args)
    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    for episode in range(args.episode_num):
        obs = env.reset()[0]
        #print(obs)
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < args.random_steps:
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents} # to get experience
            else:
                action = maddpg.select_action(obs) #using policy

            next_obs, reward, done, _, info = env.step(action)
            # /opt/anaconda3/envs/pettingzoo_m4/lib/python3.11/site-packages/pettingzoo/utils/conversions.py 
            # step function from class aec_to_parallel_wrapper
            # print('next_obs',next_obs)
            # print('reward',reward)
            # print(1)
            # env.render()
            maddpg.add(obs, action, reward, next_obs, done) #add experience to buffer
            
            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)

            obs = next_obs

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r
            

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
                # print(agent_id, r, sum_reward)
                wandb.log({'Agent':agent_id, 'curr_reward': r, 'total_Reward':sum_reward})
            message += f'sum reward: {sum_reward}'
            # wandb.log({'Reward':sum_reward})
            print(message)

    maddpg.save(episode_rewards)  # save model


    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward


    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id,alpha=0.2)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {args.env_name}'
    ax.set_title(title)
    
    plt.savefig(os.path.join(result_dir, title))
    # plt.show()
