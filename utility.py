import json
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="hangzhou_1x1_bc-tyc_18041607_1h")
    parser.add_argument("--num_step", type=int, default=3600)
    return parser.parse_args()

def parse_roadnet(roadnetFile):
    roadnet = json.load(open(roadnetFile))
    lane_phase_info_dict = {}

    for intersection in roadnet["intersections"]:
        if intersection['virtual']:
            continue
        lane_phase_info_dict[intersection['id']] = {"start_lane": [],
                                                    "end_lane": [],
                                                    "phase": [],
                                                    "lane_mapping": {},
                                                    "phase_startLane_mapping": {},
                                                    "phase_roadLink_mapping": {}}
        road_links = intersection["roadLinks"]

        start_lane = []
        end_lane = []
        roadLink_lane_pair = {}
        rik = 0
        for r in road_links:
            if r['type'] != 'turn_right':
                roadLink_lane_pair[rik] = []
                rik += 1

        rik = 0
        for ri in range(len(road_links)):
            road_link = road_links[ri]
            for lane_link in road_link["laneLinks"]:
                sl = road_link['startRoad'] + "_" + str(lane_link["startLaneIndex"])
                el = road_link['endRoad'] + "_" + str(lane_link["endLaneIndex"])
                start_lane.append(sl)
                end_lane.append(el)
                if road_link['type'] != 'turn_right':
                    roadLink_lane_pair[rik].append([sl, el])

            if road_link['type'] != 'turn_right':
                roadLink_lane_pair[rik].append(road_link['direction'])
                rik += 1

        lane_phase_info_dict[intersection['id']]["start_lane"] = sorted(list(set(start_lane)))
        lane_phase_info_dict[intersection['id']]["end_lane"] = sorted(list(set(end_lane)))
        lane_phase_info_dict[intersection['id']]["lane_mapping"] = roadLink_lane_pair

        for phase_i in range(1, 5):
            lane_phase_info_dict[intersection['id']]["phase"].append(phase_i)

    return lane_phase_info_dict


def plot_data_lists(data_list,
                    label_list,
                    length=10,
                    height=6,
                    x_label='x',
                    y_label='y',
                    label_fsize=14,
                    save=True,
                    figure_name='temp'):
    import matplotlib
    if save:
        matplotlib.use('PDF')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(length, height))
    ax.grid(True)

    for data, label in zip(data_list, label_list):
        ax.plot(data, label=label)  # mec='none', ms=3, label='Algo1 $\\lambda=0.005$'

    ax.plot()
    ax.set_xlabel(x_label, fontsize=label_fsize)
    ax.set_ylabel(y_label, fontsize=label_fsize)
    ax.legend()
    ax.grid(True)

    if save:
        plt.savefig(figure_name)
    else:
        plt.show()


def sample_aug(state):
    state_ = {}
    for id_ in state:
        temp = [e for e in state[str(id_)][0]]
        tt = temp[3]
        for i in range(3, 0, -1):
            temp[i] = temp[i - 1]
        temp[0] = tt
        temp[4] = (temp[4] % 4) + 1

        ttemp = np.array(temp)
        state_size = len(ttemp.flatten())
        ttemp = np.reshape(ttemp, [1, state_size])
        state_[str(id_)] = ttemp

    return state_

def sample_aug_single(state):

    temp = [e for e in state[0]]
    tt = temp[3]
    for i in range(3, 0, -1):
        temp[i] = temp[i - 1]
    temp[0] = tt
    temp[4] = (temp[4] % 4) + 1

    state_ = np.array(temp)
    state_size = len(state_.flatten())
    state_ = np.reshape(state_, [1, state_size])

    return state_

def sample_aug_timing(state):

    temp = [e for e in state[0]]
    tt = temp[4]
    for i in range(4, 1, -1):
        temp[i] = temp[i - 1]
    temp[1] = tt
    temp[5] = (temp[5] % 4) + 1

    state_ = np.array(temp)
    state_size = len(state_.flatten())
    state_ = np.reshape(state_, [1, state_size])

    return state_