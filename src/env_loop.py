import os
import csv
import time
from copy import deepcopy
import pickle
from gymnasium import Env
from typing import Union, List
import numpy as np
from agent.rsac import RecurrentSAC
from agent.recurrent_replay_buffer import RecurrentReplayBuffer
from env.model_variable import ModelVariable

from env.house2 import House2Env

import util.logger as logger
from util.log_helpers import record_param, record_dict


def rescale_obs(mvars:List[ModelVariable], values:List[float]) -> List[float]:
    rescaled_obs = [mvar.rescale(val) for mvar, val in zip(mvars, values)]
    return np.array(rescaled_obs)


def test_current_algorithm(
    env:House2Env, 
    algorithm:RecurrentSAC=None, 
    recurrent:bool=False, 
    std_ctr:bool=False) -> tuple:
    """Runs the algorithm with the given environment and returns observations, actions and results.
    """
    assert std_ctr or algorithm, "Either standard controller must be active or an algorithm must be given."
    
    state, _ = env.reset()
    episode_return = 0
    episode_len = 0
    done = False

    if recurrent: algorithm.reinitialize_hidden()

    while not done:
        
        if std_ctr:
            state, reward, done, _, info = env.standard_control_step()
        else:
            action = algorithm.act(state, deterministic=True)
            state, reward, done, _, info = env.step(action)
        
        new_state_ts = env.get_observation_details()
        action_details = info.pop("action")
        final_res = info.pop("final_resources", None)
        
        # Save observation as lists
        if not 'state_ts' in locals():
            state_ts = new_state_ts
            
            for p, v in zip(env.action_vars, action_details):
                state_ts[p.name] = [v for i in range(len(new_state_ts[list(new_state_ts.keys())[0]]))]
        else:
            for key in new_state_ts.keys():
                # The first observation details value equals the last observation details value
                state_ts[key].extend(new_state_ts[key][1:])
                
            for p, v in zip(env.action_vars, action_details):
                state_ts[p.name].extend([v for i in range(len(new_state_ts[list(new_state_ts.keys())[0]]) - 1)])

        # Save observation and reward as lists
        if not "obs_and_rew" in locals():
            obs_and_rew = {}
            for p, v in zip(env.observation_vars, state):
                obs_and_rew[f"{p.name}_{p.unit}"] = [v]
            for k, v in info.items():
                obs_and_rew[k] = [v]
            for p, v in zip(env.action_vars, action_details):
                obs_and_rew[p.name] = [v]
        else:
            for p, v in zip(env.observation_vars, state):
                obs_and_rew[f"{p.name}_{p.unit}"].append(v)
            for k, v in info.items():
                obs_and_rew[k].append(v)
            for p, v in zip(env.action_vars, action_details):
                obs_and_rew[p.name].append(v)

        episode_return += reward
        episode_len += 1
    return episode_len, episode_return, final_res, state_ts, obs_and_rew


def test_and_report(env:House2Env, runs:int, epoch:int, algo:RecurrentSAC=None, recurrent:bool=False, std_ctr:bool=False):
    
    episode_lens, episode_rets = [], []
    constraint_violations, electricity_costs = [], []
    eval_csv = os.path.join(logger.log_dir, "eval", "evaluation_results.csv")
    write_header = not os.path.exists(eval_csv)
    state_ts_dir = os.path.join(logger.log_dir, "eval", "state_ts")
    if not os.path.exists(state_ts_dir):
            os.makedirs(state_ts_dir)
    env_name = "Test-Env"
    
    for j in range(runs):

        episode_len, episode_ret, resources, state_ts, obs_and_res = test_current_algorithm(
            env=env, algorithm=algo, recurrent=recurrent, std_ctr=std_ctr)
        resources["name"] = env_name
        resources["return"] = episode_ret
        resources["env_steps"] = episode_len
        resources["training_round"] = epoch
        resources["episode"] = j
        
        fieldnames = resources.keys()
        with open(eval_csv, 'a', encoding='UTF8', newline='') as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_header:
                csv_writer.writeheader()
                write_header = False
            
            csv_writer.writerow(resources)
        
        fieldnames = sorted(state_ts.keys())
        state_ts_csv = os.path.join(state_ts_dir, f"{epoch}_{j}_{env_name}_details.csv")
        
        with open(state_ts_csv, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            writer.writerows(zip(*[state_ts[key] for key in fieldnames]))
        
        fieldnames = obs_and_res.keys()
        obs_and_res_csv = os.path.join(state_ts_dir, f"{epoch}_{j}_{env_name}_o_r.csv")
        
        with open(obs_and_res_csv, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            writer.writerows(zip(*[obs_and_res[key] for key in fieldnames]))
        
        constraint_violations.append(resources["Constraint violations"])
        electricity_costs.append(resources["Total electricity costs"])
            
        episode_lens.append(episode_len)
        episode_rets.append(episode_ret)
    
    return episode_lens, episode_rets, constraint_violations, electricity_costs


def train(
        env:Env,
        algorithm: RecurrentSAC,
        buffer: RecurrentReplayBuffer,
        num_epochs:int,
        num_steps_per_epoch:int,
        num_test_episodes_per_epoch:int,
        update_after:int,
        test_env:Env=None,
        update_every:int=1,
        actor_save_dir:str='',
        buffer_save_file:str='',
) -> None:
    """
    Function containing the main loop for environment interaction / learning / testing.
    Args:
        env: Training environment.
        algorithm: Algorithm for training.
        buffer: Replay buffer for training.
        num_epochs: Number of epochs to train for.
        num_steps_per_epoch: Number of environment steps per epoch.
        num_test_episodes_per_epoch: Number of test episodes after each epoch.
        update_after: Number of environment steps before training starts.
        test_env: Test environment.
        update_every: Number of environment steps between training updates.
        actor_save_dir: Directory to save the actor to. If empty, the actor will not be saved.
        buffer_save_file: Directory to save the replay buffer to. If empty, the buffer will not be saved.
    """


    # prepare stats trackers
    episode_len = 0
    episode_ret = 0
    train_episode_lens = []
    train_episode_rets = []
    algo_specific_stats_tracker = []
    train_done_eps_len = []
    train_done_eps_ret = []
    train_constraint_violations = []
    train_electricity_costs = []
    
    start_time = time.perf_counter()

    # @@@@@@@@@@ training loop @@@@@@@@@@

    state, _ = env.reset()
    save_buffer = True

    # Since algorithm is a recurrent policy, it (ideally) shouldn't be updated during an episode since this would
    # affect its ability to interpret past hidden states. Therefore, during an episode, algorithm_clone is updated
    # while algorithm is not. Once an episode has finished, we do algorithm.copy_networks_from(algorithm_clone) to
    # carry over the changes.

    algorithm_clone = deepcopy(algorithm)  # algorithm is for action; algorithm_clone is for updates and testing

    for t in range(num_steps_per_epoch * num_epochs):

        # @@@@@@@@@@ environment interaction @@@@@@@@@@

        if t >= update_after:
            # exploration is done
            if buffer_save_file and save_buffer:
                # Save the buffer once it is filled with random environment transitions. Helps with debugging.
                with open(buffer_save_file, 'wb') as outp:
                    pickle.dump(buffer, outp, pickle.HIGHEST_PROTOCOL)
                logger.info("Buffer saved.")
                save_buffer = False
            action = algorithm.act(state, deterministic=False)
        else:
            action = env.action_space.sample()

        next_state, reward, done, _, info = env.step(action)
        episode_len += 1
        
        # Log and save values
        final_res = info.pop("final_resources", None)
        record_param(env.observation_vars, next_state, "state", t)
        record_param(env.observation_vars, rescale_obs(env.observation_vars, state), "rescaled state", t)
        record_param(env.action_vars, info.pop("action"), "action", t)
        record_dict(info, "reward", t)

        episode_ret += reward
        buffer.push(state, action, reward, next_state, done)
        state = next_state

        # @@@@@@@@@@ end of trajectory handling @@@@@@@@@@

        if done:
            
            logger.record("eps_len", episode_len, t)
            logger.record("eps_ret", episode_ret, t)

            train_episode_lens.append(episode_len)
            train_episode_rets.append(episode_ret)
            
            if final_res is not None:
                train_done_eps_len.append(episode_len)
                train_done_eps_ret.append(episode_ret)
                train_constraint_violations.append(final_res["Constraint violations"])
                train_electricity_costs.append(final_res["Total electricity costs"])
            
            # reset environment and stats trackers
            (state, _), episode_len, episode_ret = env.reset(), 0, 0

            algorithm.copy_networks_from(algorithm_clone)
            algorithm.reinitialize_hidden()  # crucial, crucial step for recurrent agents
        

        # @@@@@@@@@@ update handling @@@@@@@@@@

        if t >= update_after and (t + 1) % update_every == 0:
            for j in range(update_every):
                batch = buffer.sample()
                algo_specific_stats = algorithm_clone.update_networks(batch)
                algo_specific_stats_tracker.append(algo_specific_stats)

        # @@@@@@@@@@ end of epoch handling @@@@@@@@@@

        if (t + 1) % num_steps_per_epoch == 0:

            epoch = (t + 1) // num_steps_per_epoch

            # @@@@@@@@@@ algo specific stats (averaged across update steps) @@@@@@@@@@

            algo_specific_stats_over_epoch = {}

            if len(algo_specific_stats_tracker) != 0:
                # get keys from the first one; all dicts SHOULD share the same keys
                keys = algo_specific_stats_tracker[0].keys()
                for k in keys:
                    values = []
                    for dictionary in algo_specific_stats_tracker:
                        values.append(dictionary[k])
                    algo_specific_stats_over_epoch[k] = np.mean(values)
                algo_specific_stats_tracker = []

            # @@@@@@@@@@ training stats (averaged across episodes) @@@@@@@@@@

            mean_train_episode_len = float(np.mean(train_episode_lens)) if train_episode_lens else np.nan
            mean_train_episode_ret = float(np.mean(train_episode_rets)) if train_episode_rets else np.nan
            mean_train_done_eps_len = float(np.mean(train_done_eps_len)) if train_done_eps_len else np.nan
            mean_train_done_eps_ret = float(np.mean(train_done_eps_ret)) if train_done_eps_ret else np.nan
            mean_train_constraint_violations = float(np.mean(train_constraint_violations)) if train_constraint_violations else np.nan
            mean_train_electricity_costs = float(np.mean(train_electricity_costs)) if train_electricity_costs else np.nan
            
            train_episode_lens = []
            train_episode_rets = []
            train_done_eps_len = []
            train_done_eps_ret = []
            train_constraint_violations = []
            train_electricity_costs = []

             # @@@@@@@@@@ testing stats (averaged across episodes) @@@@@@@@@@

            # testing may happen during the middle of an episode, and hence "algorithm" may not contain the latest parameters
            test_algorithm = deepcopy(algorithm)
            test_algorithm.copy_networks_from(algorithm_clone)
            # test_algorithm = deepcopy(algorithm_clone)
            
            if actor_save_dir:
                epoch_save_dir = os.path.join(actor_save_dir, str(epoch))
                if not os.path.exists(epoch_save_dir):
                    os.makedirs(epoch_save_dir)
                algorithm.save_actor(epoch_save_dir)

            test_episode_lens, test_episode_rets, test_constraint_violations, test_electricity_costs = test_and_report(
                env=test_env, algo=test_algorithm, runs=num_test_episodes_per_epoch,
                epoch=epoch, recurrent=True, std_ctr=False)

            mean_test_episode_len = float(np.mean(test_episode_lens))
            mean_test_episode_ret = float(np.mean(test_episode_rets))
            mean_test_constraint_violations = float(np.mean(test_constraint_violations))
            mean_test_electricity_costs = float(np.mean(test_electricity_costs))

            # @@@@@@@@@@ hours elapsed @@@@@@@@@@

            current_time = time.perf_counter()
            hours_elapsed = (current_time - start_time) / 60 / 60

            # @@@@@@@@@@ wandb logging @@@@@@@@@@

            record_dict({
                'Episode Length (Train)': mean_train_episode_len,
                'Episode Return (Train)': mean_train_episode_ret,
                'Full episode Length (Train)': mean_train_done_eps_len,
                'Full episode Return (Train)': mean_train_done_eps_ret,
                'Full episode Constraint Violations (Train)': mean_train_constraint_violations,
                'Full episode Electricity Costs (Train)': mean_train_electricity_costs,
                'Episode Length (Test)': mean_test_episode_len,
                'Episode Return (Test)': mean_test_episode_ret,
                'Episode Constraint Violations (Test)': mean_test_constraint_violations,
                'Episode Electricity Costs (Test)': mean_test_electricity_costs,
                'Hours': hours_elapsed
            }, "episode", t)

            # @@@@@@@@@@ console logging @@@@@@@@@@

            stats_string = (
                f"===============================================================\n"
                f"| Epochs                  | {epoch}/{num_epochs}\n"
                f"| Timesteps               | {t+1}\n"
                f"| Episode Length (Train)  | {round(mean_train_episode_len, 2)}\n"
                f"| Episode Return (Train)  | {round(mean_train_episode_ret, 2)}\n"
                f"| Episode Length (Test)   | {round(mean_test_episode_len, 2)}\n"
                f"| Episode Return (Test)   | {round(mean_test_episode_ret, 2)}\n"
                f"| Hours                   | {round(hours_elapsed, 2)}\n"
                f"==============================================================="
            )

            logger.info(stats_string)

    algorithm.save_actor(actor_save_dir) 
    algorithm.save_Q(actor_save_dir)