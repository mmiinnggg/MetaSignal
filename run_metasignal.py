import math
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

import argparse
import logging
from datetime import datetime
from time import *
import csv

from cityflow_env import CityFlowEnvM
from environment import TSCEnv
from agent import TOSFB
from metric import TravelTimeMetric
from utility import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--thread', type=int, default=8, help='number of threads')
parser.add_argument('--num_step', type=int, default=3600, help='number of steps')
parser.add_argument('--phase_step', type=int, default=10, help='Minimum phase duration')
parser.add_argument('--save_model', action="store_true", default=False)
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--epoch', type=int, default=80, help='training episodes')
parser.add_argument("--save_rate", type=int, default=20, help="save model once every time this many episodes are completed")
parser.add_argument('--config_file', type=str, default='', help='path of config file')
parser.add_argument('--test_flow_floder', type=str, default='', help='path of config file')
parser.add_argument('--save_dir', type=str, default="", help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="", help='directory in which logs should be saved')
args = parser.parse_args()

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
crt_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_name = args.log_dir + '/' + args.mode + '_' + args.dataset + '_' + 'Intosfb' + '_' + str(args.epoch)+  '_' +  crt_time + '.csv'
CsvFile = open(log_name, 'w')
CsvWriter = csv.writer(CsvFile)
CsvWriter.writerow(
    ["Mode", "episode", "step", "travel_time", "throughput", "speed score", "max waiting", "mean_episode_reward", "mean_td_error"])
CsvFile.close()

def build(path = args.config_file):

    with open(path) as f:
        cityflow_config = json.load(f)

    config = {}
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)

    intersection_id = list(config['lane_phase_info'].keys()) 
    config["intersection_id"] = intersection_id
    phase_list = {id_: config["lane_phase_info"][id_]["phase"] for id_ in intersection_id}
    config["phase_list"] = phase_list

    timing_list = {id_: [i*5 + args.phase_step for i in range(1,5)] for id_ in intersection_id}
    config['timing_list'] = timing_list

    world = CityFlowEnvM(config["lane_phase_info"],
                       intersection_id,
                       num_step=args.num_step,
                       thread_num=args.thread,
                       cityflow_config_file=args.config_file,
                       dataset=args.dataset
                       )
    print("world built.")

    config["state_size"] = world.state_size
    agents = {}
    for id_ in intersection_id:
        agent = TOSFB(intersection_id = id_,
                      state_size = config["state_size"],
                      action_size = len(phase_list[id_]),
                      batch_size = args.batch_size,
                      phase_list = phase_list[id_],
                      timing_list = timing_list[id_],
                      env = world)
        agents[id_] = agent
    print("agents built.")

    metrics = [TravelTimeMetric(world)]

    return config, world, agents, metrics

def train(path = args.config_file):
    config, world, agents, metrics = build(path)

    print("training processing...")

    EPISODES = args.epoch
    episode_travel_time = []
    total_decision_num = {id_: 0 for id_ in config["intersection_id"]}

    total_step = 0
    with tqdm(total=EPISODES * args.num_step / args.action_interval) as pbar:
        for e in range(EPISODES):
            action = {}
            action_phase = {}
            timing_phase = {}
            simulation_time = 0
            reward = {id_: 0 for id_ in config["intersection_id"]}
            rest_timing = {id_: 0 for id_ in config["intersection_id"]}

            episodes_decision_num = {id_: 0 for id_ in config["intersection_id"]}
            episodes_rewards = {id_: 0 for id_ in config["intersection_id"]}
            td_errors = {id_: 0 for id_ in config["intersection_id"]}

            world.reset()
            for metric in metrics:
                metric.reset()
            state = {}
            for id_ in config["intersection_id"]:
                agents[id_].reset_traces()
                state[id_] = world.get_state_(id_)

            i = 0
            while i < args.num_step:
                if i % args.action_interval == 0:
                    for id_ in config["intersection_id"]:
                        if total_decision_num[id_] > agents[id_].learning_start:
                            action[id_] = agents[id_].choose_action(state[id_])
                        else:
                            action[id_] = agents[id_].sample()
                        action_phase[id_] = config["phase_list"][id_][action[id_]]

                    for _ in range(args.action_interval):
                        next_state, reward_, t1 = world.step(action_phase, i)  
                        i += 1
                        if world.eng.get_current_time() % 5 == 0:
                            for metric in metrics:
                                metric.update()
                    simulation_time += t1

                    for id_ in config["intersection_id"]:
                        next_state[id_] = np.array(next_state[id_], dtype=np.float32) * 0.01
                        reward[id_] = world.get_reward_(id_)
                        agents[id_].remember(state[id_], action_phase[id_], reward[id_], next_state[id_])
                        total_decision_num[id_] += 1
                        episodes_decision_num[id_] += 1
                        episodes_rewards[id_] += reward[id_]
                        td_errors[id_] += agents[id_].td_error
                        state[id_] = next_state[id_]

                total_step += 1
                pbar.update(1)
                print_reward = deepcopy(reward)
                pbar.set_description(
                    "t_st:{}, epi:{}, st:{}, r:{} ".format(total_step, e+1, i+1, print_reward))

            if e % args.save_rate == args.save_rate - 1:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                for id_ in config["intersection_id"]:
                    agents[id_].save_model(args.save_dir, e)

            episode_travel_time.append(world.eng.get_average_travel_time())
            print('\n Epoch {} travel time:'.format(e+1), world.eng.get_average_travel_time())
            for metric in metrics:
                print(f"\t{metric.name}: {metric.eval()}")

            mean_reward = {id_: [] for id_ in config["intersection_id"]}
            mean_td_error = {id_: [] for id_ in config["intersection_id"]}
            for id_ in config["intersection_id"]:
                mean_reward[id_] = episodes_rewards[id_] / episodes_decision_num[id_]
                mean_td_error[id_] = td_errors[id_] / episodes_decision_num[id_]

            CsvFile = open(log_name, 'a+')
            CsvWriter = csv.writer(CsvFile)
            CsvWriter.writerow(
                ["-", e+1, i+1, metrics[0].eval(), metrics[1].eval(), metrics[2].eval(), metrics[3].eval(),
                 np.mean(list(mean_reward.values())), np.mean(list(mean_td_error.values()))   ])
            CsvFile.close()

    plot_data_lists([episode_travel_time], ['travel time'], figure_name=args.log_dir + '/'+ str(args.epoch)+ '_' + args.dataset + '_' +  crt_time +'_travel time.pdf')


def test(path = args.config_file):
    config, world, agents, metrics = build(path)
    print("testing processing...")

    total_step = 0
    with tqdm(total= args.num_step) as pbar:
        action = {}
        action_phase = {}
        timing_phase = {}
        reward = {id_: 0 for id_ in config["intersection_id"]}
        rest_timing = {id_: 0 for id_ in config["intersection_id"]}

        episodes_decision_num = {id_: 0 for id_ in config["intersection_id"]}
        episodes_rewards = {id_: 0 for id_ in config["intersection_id"]}
        td_errors = {id_: 0 for id_ in config["intersection_id"]}

        world.reset()
        for metric in metrics:
            metric.reset()
        state = {}
        for id_ in config["intersection_id"]:
            state[id_] = world.get_state_(id_)
            agents[id_].load_model(args.save_dir, args.epoch - 1)
        print("agents loaded...")

        for i in range(args.num_step):
            for id_, t in rest_timing.items():
                if t == 0:
                    if i != 0:
                        reward[id_] = world.get_reward_(id_)
                        episodes_decision_num[id_] += 1
                        episodes_rewards[id_] += reward[id_]
                        td_errors[id_] += agents[id_].td_error
                        state[id_] = next_state[id_]

                    action[id_] = agents[id_].choose_action(state[id_])
                    action_phase[id_] = config["phase_list"][id_][action[id_]]

                    p, timing_phase[id_] = world.get_timing_(id_, action_phase[id_])
                    rest_timing[id_] = timing_phase[id_]

            next_state, reward_, t1 = world.step(action_phase, i)  
            if world.eng.get_current_time() % 5 == 0:
                for metric in metrics:
                    metric.update()
            for id_ in rest_timing:
                rest_timing[id_] -= 1
                next_state[id_] = np.array(next_state[id_], dtype=np.float32) * 0.01

            total_step += 1
            pbar.update(1)
            print_reward = deepcopy(reward)
            pbar.set_description(
                "t_st:{}, epi:{}, st:{}, r:{} ".format(total_step, 0, i+1, print_reward))

        print('\n Test Epoch {} travel time:'.format(0), world.eng.get_average_travel_time())
        for metric in metrics:
            print(f"\t{metric.name}: {metric.eval()}")

        mean_reward = {id_: [] for id_ in config["intersection_id"]}
        mean_td_error = {id_: [] for id_ in config["intersection_id"]}
        for id_ in config["intersection_id"]:
            mean_reward[id_] = episodes_rewards[id_] / episodes_decision_num[id_]
            mean_td_error[id_] = td_errors[id_] / episodes_decision_num[id_]

        CsvFile = open(log_name, 'a+')
        CsvWriter = csv.writer(CsvFile)
        CsvWriter.writerow(
            ["test", "-", i+1, metrics[0].eval(), metrics[1].eval(), metrics[2].eval(), metrics[3].eval(),
             np.mean(list(mean_reward.values())), np.mean(list(mean_td_error.values()))  ])
        CsvFile.close()

    return world.eng.get_average_travel_time()

def meta_test(path = args.config_file):
    config, world, agents, metrics = build(path)
    print("meta-testing processing...")

    total_decision_num = {id_: 0 for id_ in config["intersection_id"]}
    total_step = 0
    with tqdm(total= args.num_step) as pbar:
        action = {}
        action_phase = {}
        timing_phase = {}
        reward = {id_: 0 for id_ in config["intersection_id"]}
        rest_timing = {id_: 0 for id_ in config["intersection_id"]}

        episodes_decision_num = {id_: 0 for id_ in config["intersection_id"]}
        episodes_rewards = {id_: 0 for id_ in config["intersection_id"]}
        td_errors = {id_: 0 for id_ in config["intersection_id"]}

        world.reset()
        for metric in metrics:
            metric.reset()
        state = {}
        for id_ in config["intersection_id"]:
            state[id_] = world.get_state_(id_)
            agents[id_].load_model(args.save_dir, args.epoch - 1)
        print("agents loaded...")

        for i in range(args.num_step):
            for id_, t in rest_timing.items():
                if t == 0:
                    action[id_] = agents[id_].choose_action(state[id_])
                    action_phase[id_] = config["phase_list"][id_][action[id_]]

                    p, timing_phase[id_] = world.get_timing_(id_, action_phase[id_])
                    rest_timing[id_] = timing_phase[id_]

            next_state, reward_, t1 = world.step(action_phase, i) 
            if world.eng.get_current_time() % 5 == 0:
                for metric in metrics:
                    metric.update()
            for id_ in rest_timing:
                rest_timing[id_] -= 1
                next_state[id_] = np.array(next_state[id_], dtype=np.float32) * 0.01

            for id_ in config["intersection_id"]:
                reward[id_] = world.get_reward_(id_)
                agents[id_].remember(state[id_], action_phase[id_], reward[id_], next_state[id_])
                total_decision_num[id_] += 1
                episodes_decision_num[id_] += 1
                episodes_rewards[id_] += reward[id_]
                td_errors[id_] += agents[id_].td_error
                state[id_] = next_state[id_]

            total_step += 1
            pbar.update(1)
            print_reward = deepcopy(reward)
            pbar.set_description(
                "t_st:{}, epi:{}, st:{}, r:{} ".format(total_step, 0, i+1, print_reward))

        t1 = world.eng.get_average_travel_time()
        print('\n Meta-Test Epoch {} total_decision_num:'.format(0), total_decision_num)
        for metric in metrics:
            print(f"\t{metric.name}: {metric.eval()}")

        mean_reward = {id_: [] for id_ in config["intersection_id"]}
        mean_td_error = {id_: [] for id_ in config["intersection_id"]}
        for id_ in config["intersection_id"]:
            mean_reward[id_] = episodes_rewards[id_] / episodes_decision_num[id_]
            mean_td_error[id_] = td_errors[id_] / episodes_decision_num[id_]

        CsvFile = open(log_name, 'a+')
        CsvWriter = csv.writer(CsvFile)
        CsvWriter.writerow(
            ["META-test", "-", i+1, metrics[0].eval(), metrics[1].eval(), metrics[2].eval(), metrics[3].eval(),
             np.mean(list(mean_reward.values())), np.mean(list(mean_td_error.values()))  ])
        CsvFile.close()

    print("testing processing...")

    total_decision_num = {id_: 0 for id_ in config["intersection_id"]}
    total_step = 0
    with tqdm(total=args.num_step) as pbar:
        action = {}
        action_phase = {}
        timing_phase = {}
        reward = {id_: 0 for id_ in config["intersection_id"]}
        rest_timing = {id_: 0 for id_ in config["intersection_id"]}

        episodes_decision_num = {id_: 0 for id_ in config["intersection_id"]}
        episodes_rewards = {id_: 0 for id_ in config["intersection_id"]}
        td_errors = {id_: 0 for id_ in config["intersection_id"]}

        world.reset()
        for metric in metrics:
            metric.reset()
        state = {}
        for id_ in config["intersection_id"]:
            state[id_] = world.get_state_(id_)
            agents[id_].reset_traces()

        for i in range(args.num_step):
            for id_, t in rest_timing.items():
                if t == 0:
                    if i != 0:
                        reward[id_] = world.get_reward_(id_)
                        total_decision_num[id_] += 1
                        episodes_decision_num[id_] += 1
                        episodes_rewards[id_] += reward[id_]
                        td_errors[id_] += agents[id_].td_error
                        state[id_] = next_state[id_]

                    action[id_] = agents[id_].choose_action(state[id_])
                    action_phase[id_] = config["phase_list"][id_][action[id_]]

                    p, timing_phase[id_] = world.get_timing_(id_, action_phase[id_])
                    rest_timing[id_] = timing_phase[id_]

            next_state, reward_, t1 = world.step(action_phase, i)  
            if world.eng.get_current_time() % 5 == 0:
                for metric in metrics:
                    metric.update()
            for id_ in rest_timing:
                rest_timing[id_] -= 1
                next_state[id_] = np.array(next_state[id_], dtype=np.float32) * 0.01

            total_step += 1
            pbar.update(1)
            print_reward = deepcopy(reward)
            pbar.set_description(
                "t_st:{}, epi:{}, st:{}, r:{} ".format(total_step, 0, i+1, print_reward))

        t2 = world.eng.get_average_travel_time()
        print('\n Test Epoch {} total_decision_num:'.format(0), total_decision_num)
        for metric in metrics:
            print(f"\t{metric.name}: {metric.eval()}")

        mean_reward = {id_: [] for id_ in config["intersection_id"]}
        mean_td_error = {id_: [] for id_ in config["intersection_id"]}
        for id_ in config["intersection_id"]:
            mean_reward[id_] = episodes_rewards[id_] / episodes_decision_num[id_]
            mean_td_error[id_] = td_errors[id_] / episodes_decision_num[id_]

        CsvFile = open(log_name, 'a+')
        CsvWriter = csv.writer(CsvFile)
        CsvWriter.writerow(
            ["meta-test", "-", i + 1, metrics[0].eval(), metrics[1].eval(), metrics[2].eval(),
             metrics[3].eval(),
             np.mean(list(mean_reward.values())), np.mean(list(mean_td_error.values()))])
        CsvFile.close()

    return t1

if __name__ == '__main__':
    start_time = time()

    train()

    test_flow_path = []
    for root, dirs, files in os.walk(args.test_flow_floder):
        for file in files:
            test_flow_path.append(args.test_flow_floder + file)
    print("Meta Test Fake")
    result = []
    for n in range(len(test_flow_path)):
        print("Meta Test Env: %d" % n)
        CsvFile = open(log_name, 'a+')
        CsvWriter = csv.writer(CsvFile)
        CsvWriter.writerow(
            [test_flow_path[n]])
        CsvFile.close()
        t1 = test(test_flow_path[n])
        t2 = meta_test(test_flow_path[n])

