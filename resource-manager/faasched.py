from cmath import inf
from faulthandler import disable
from pydoc import doc
from queue import PriorityQueue
from sqlite3 import Timestamp
import numpy as np
from collections import deque

import os
import random
import sys
import time
import math

from os import listdir
from os.path import isfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from sklearn import preprocessing

from statistics import mean
from random import randrange
import csv

FAAS_ROOT="/home/abhisek/Serverless-SLA/faas-profiler"
sys.path = [FAAS_ROOT, FAAS_ROOT+'/database'] + sys.path
from ContactDB import GetCountLSRecordsSince
from ContactDB import FlushActivation

cpu_layout = ["0,16", "1,17", "2,18", "3,19", "4,20", "5,21", "6,22", "7,23", "8,24", "9,25", "10,26", "11,27", "12,28", "13,29", "14,30", "15,31"]

# Fill the following dictionaries with the application data
configuration = {
    'faasched:dummy': {'name': 'dummy', 'min_cpu_wait': 25, 'max_cpu_wait': 50, 'ipc': 1.3, 'min_l1d_mpki': 5.99, 'max_l1d_mpki': 6.34, 'min_l2_mpki': 0.06, 'max_l2_mpki': 0.16, 'min_nvcs': 0, 'max_nvcs': 100, 'instruction': 665, "SPI": 4.539063182103345e-10},}

ls_apps = ["binary_alert", "squeezenet", "email_gen", "markdown2html", "stock_analysis"]

def ExtractExtraAnnotations(json_annotations_data):
    """
    Extracts deep information from activation json record.
    """
    extra_data = {'waitTime': [], 'initTime': [], 'kind': []}

    for item in json_annotations_data:
        if item['key'] in extra_data.keys():
            extra_data[item['key']] = item['value']

    for key in extra_data.keys():
        if extra_data[key] == []:
            extra_data[key] = 0

    return extra_data

"""
    Policy Class:
    Creates a NN network for the policy gradient reinforcement learning technique.

    params: s_size is input_dimension
            a_size is output dimension
            h_size is hidden layer size
"""
class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Actor, self).__init__()

        self.num_actions = num_actions
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return policy_dist

class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Critic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, state):
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        return value

class ServerlessAgent:
    def __init__(self, sperf_weight, sfair_weight, tau) -> None:
        hidden_size = 200
        learning_rate = 1e-4
        state_space = 13
        action_space = 12

        self.sperf_weight  = sperf_weight
        self.sfair_weight = sfair_weight
        self.tau = tau

        # LS Applications
        self.app = ['bs', 'eg', 'mr', 'sa', 'od']
        
        self.prio_step_size = 10
        self.affin_step_size = 1
        self.epsilon = 0.1

        # Fill the following dictionaries with the application data

        # The state is defined by the following parameters:
        # 1. L1DMPKI
        # 2. L2MPKI
        # 3. IPC
        # 4. CPU wait time
        # 5. nvcs events
        # 6. Resource fairness
        # 7. Priority
        # 8. Affinity
        # 9. Employ futex lock
        # 10. P_low
        # 11. P_high
        # 12. A_other
        # 13. #Containers
        self.data = {
            'dummy': {'priority': 50, "affin": 5, 'cpus': '', 'policy_dist': [], 'action': -1, 'value': 0, 'state': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'cur_cpu_state': {'cpu_wait_time': 0, 'nvcs': 0}, 'prev_cpu_state': {'cpu_wait_time': 0, 'nvcs': 0}, 'reward': 0, 'total_affin': 0}}

        self.last_cpu = 0

        self.S_cpu = 0

        self.ls_image = ""
        self.ld_image = ""

        self.agent_policy = Actor(state_space, action_space, hidden_size)
        self.agent_optimizer = optim.Adam(self.agent_policy.parameters(), lr=learning_rate)

        self.critic_value = Critic(state_space, 1, hidden_size)
        self.critic_optimizer = optim.Adam(self.critic_value.parameters(), lr=learning_rate)

        self.adjust_prio_command = 'sudo bash ' + FAAS_ROOT + '/state-monitor/adjust_priority.sh'
        self.state_path = os.path.join(FAAS_ROOT, "state")

    def reset(self):
        self.priority = 50

    def load(self):
        self.agent_policy.load_state_dict(torch.load(FAAS_ROOT+f"/models/faasched_model_1.0_1.0_0.5.pth"))

    def nn_state(self, train):
        if train:
            self.agent_policy.train(mode=True)
        else:
            self.agent_policy.train(mode=False)

    def increment_prio(self, priority):
        return min(80, priority + self.prio_step_size)

    def decrement_prio(self, priority):
        return max(1, priority - self.prio_step_size)

    def increment_affin(self, affin):
        if(affin == -1):
            return self.affin_step_size
        else:
            return min(16, affin + self.affin_step_size)

    def decrement_affin(self, affin):
        if(affin == -1):
            return -1
        if(affin == self.affin_step_size):
            return -1
        else:
            return max(1, affin - self.affin_step_size)

    def get_exec_details(self, since):
        activations = GetCountLSRecordsSince(since=since, limit=100000)

        dummy_num_reqs = 0

        if 'error' in activations.keys():
            print('Encountered an error getting data from the DB! Check the logs for more info.')
            return None

        activations = activations['docs']

        for activation in activations:
            if(activation["name"] == "dummy"):
                dummy_num_reqs +=1

        return dummy_num_reqs

    def get_stats(self):
        dummy_stats = {'cpu_wait_time': 0, 'nvcs': 0, 'L1-dcache-load-misses': 0, 'L1-icache-load-misses': 0, 'frontend_retired.l2_miss': 0, 'instructions': 0, 'cpu-cycles': 0}

        for f in listdir(self.state_path):
            if isfile(os.path.join(self.state_path, f)):
                if "perflog" in f:
                    with open(os.path.join(self.state_path, f)) as file:
                        lines = file.readlines()
                            
                    if ((len(lines) < 2) or ("no access to" in lines[1])):
                        print(os.path.join(self.state_path, f))
                        print(lines)

                    for line in lines:
                        separated = line.split(' ')
                        separated = [v for v in separated if v != '']
                        if(separated[0]=="#")or(len(separated) < 3):
                            continue
                     
                        field = separated[2]
                        if(field == "msec") or ("counted" in field):
                            field = separated[3] if separated[3] != "msec" else separated[4]

                        try:
                            if("not" in separated[1]):
                                val = 0
                            else:
                                val = float(separated[1].replace(',', ''))
                        except Exception as e:
                            print(e)
                            val = None

                        try:
                            if "dummy" in f:
                                dummy_stats[field] += val
                            
                        except:
                            if "dummy" in f:
                                dummy_stats[field] = val
                            
                if "perfsystime" in f:
                    with open(os.path.join(self.state_path, f)) as file:
                        lines = file.readlines()

                    pre_cpu_stat = lines[0].split(' ', 11)
                    post_cpu_stat = lines[-1].split(' ', 11)

                    try:
                        if "dummy" in f:
                            dummy_stats['nvcs'] += int(post_cpu_stat[6].strip())
                            dummy_stats['cpu_wait_time'] += int(int(post_cpu_stat[7].strip()))
                    except:
                        print(f, post_cpu_stat)
                        pass

        try:
            dummy_stats['IPC'] = dummy_stats['instructions'] / dummy_stats['cpu-cycles']
        except:
            dummy_stats['IPC'] = 0

        return dummy_stats
    
    # Get the number of containers that are running for an image
    def get_ld_info(self):
        with open(os.path.join(self.state_path, 'ld_apps_info')) as f:
            lines = f.readlines()

        running_apps = {}
        for line in lines[1:]:
            image = "NaN" if((len(line.split()) < 2)) else line.split()[1].strip()
            cntrs = 0 if((len(line.split()) < 2)) else int(line.split()[0].strip())
            running_apps[configuration[image]['name']] = cntrs

        return lines[0].strip(), running_apps

    def take_action(self, action, priority, affin):
        if action == 0:
            return self.increment_prio(priority), -1
        
        elif action == 1:
            return self.decrement_prio(priority), -1

        elif action == 2:
            return self.increment_prio(priority), self.decrement_affin(affin)

        elif action == 3:
            return self.decrement_prio(priority), self.increment_affin(affin)
        
        elif action == 4:
            return self.decrement_prio(priority), self.decrement_affin(affin)

        elif action == 5:
            return self.increment_prio(priority), self.increment_affin(affin)

        elif action == 6:
            return self.increment_prio(priority), affin

        elif action == 7:
            return self.decrement_prio(priority), affin
        
        elif action == 8:
            return priority, self.increment_affin(affin)

        elif action == 9:
            return priority, self.decrement_affin(affin)
        
        elif action == 10:
            return priority, affin
        
        elif action == 11:
            return priority, -1

    def get_policy(self, app_state, app):
        state = torch.from_numpy(np.array(app_state)).float().unsqueeze(0)

        policy_dist = self.agent_policy.forward(state)
        value_dist = self.critic_value.forward(state)
        value = value_dist.detach().numpy()[0,0]
        dist = policy_dist.detach().numpy()

        p = np.random.random()

        if p < self.epsilon:
            if(app == "od"):
                action = random.choice([0, 1, 11])
            else:
                action = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        else:
            if(app == "od"):
                actions = [0, 1, 11]
                dist_new = np.array(dist[0])[actions]
                action = actions[np.argmax(dist_new)]
            else:
                actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                dist_new = np.array(dist[0])[actions]
                action = actions[np.argmax(dist_new)]

        return policy_dist, action, value

    def train_policy(self, policy_dist, action, reward, value, app_state):
        if isinstance(policy_dist, list):
            print(policy_dist)
            policy_dist = np.array(policy_dist)
            print("Got List")
            return
        log_prob = torch.log(policy_dist.squeeze()[action])
        state = torch.from_numpy(np.array(app_state)).float().unsqueeze(0)
        value_dist = self.critic_value.forward(state)
        value_next = value_dist.detach().numpy()[0,0]

        advantage = reward + 0.99*value_next - value

        agent_loss = torch.tensor([-log_prob * advantage], requires_grad=True)
        critic_loss = torch.tensor([advantage*advantage], requires_grad=True)

        print("Agent loss:", agent_loss, "Critic loss:", critic_loss)

        self.agent_optimizer.zero_grad()
        agent_loss.backward()
        self.agent_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def get_others(self, app_name):
        p_low = 0
        p_high = 0
        a_other = 0

        for app in self.app:
            if (self.data[app]['priority'] <= self.data[app_name]['priority']):
                p_low+=1
            elif (self.data[app]['priority'] > self.data[app_name]['priority']):
                p_high+=1
            if(app != app_name):
                a_other += self.data[app]['affin'] if self.data[app]['affin'] > 0 else 0

        return p_low, p_high, a_other

    def scaled_min_max(self, x, min_x, max_x, scale):
        return ((x-min_x)/(max_x-min_x))*scale
    
    def min_max(self, x, min_x, max_x):
        return ((x-min_x)/(max_x-min_x))

    def get_state(self, app, ls_stats, ls_image, num_reqs, cnts):
        self.data[app]['cur_cpu_state']['cpu_wait_time'] = ls_stats['cpu_wait_time']
        self.data[app]['cur_cpu_state']['nvcs'] = ls_stats['nvcs']
        wait_time = 0

        if(num_reqs != 0):
            state = [0, 0, 0, 0, 0, 0]

            # 1. CPU wait time/#instructions (F_wait)
            if(self.data[app]['cur_cpu_state']['cpu_wait_time'] > self.data[app]['prev_cpu_state']['cpu_wait_time']):
                state[0] = (self.data[app]['cur_cpu_state']['cpu_wait_time'] - self.data[app]['prev_cpu_state']['cpu_wait_time']) / ls_stats['instructions'] if ls_stats['instructions'] != 0 else 0
            else:
                state[0] = self.data[app]['cur_cpu_state']['cpu_wait_time'] / ls_stats['instructions'] if ls_stats['instructions'] != 0 else 0

            wait_time = state[0]
            # state[0] = self.min_max(state[0], configuration[ls_image]['min_cpu_wait'], configuration[ls_image]['max_cpu_wait'])

            # 2. nvcs events
            if(self.data[app]['cur_cpu_state']['nvcs'] > self.data[app]['prev_cpu_state']['nvcs']):
                state[1] = (self.data[app]['cur_cpu_state']['nvcs'] - self.data[app]['prev_cpu_state']['nvcs']) / num_reqs
            else:
                state[1] = self.data[app]['cur_cpu_state']['nvcs']  / num_reqs

            state[1] = self.min_max(state[1], configuration[ls_image]['min_nvcs'], configuration[ls_image]['max_nvcs'])

            # 3. L1DMPKI
            state[2] = (ls_stats['L1-dcache-load-misses']*1000) / ls_stats['instructions'] if ls_stats['instructions'] != 0 else 0
            state[2] = self.min_max(state[2], configuration[ls_image]['min_l1d_mpki'], configuration[ls_image]['max_l1d_mpki'])

            # 4. L2MPKI
            state[3] = (ls_stats['frontend_retired.l2_miss']*1000) / ls_stats['instructions'] if ls_stats['instructions'] != 0 else 0
            state[3] = self.min_max(state[3], configuration[ls_image]['min_l2_mpki'], configuration[ls_image]['max_l2_mpki'])

            # 5. IPC
            state[4] = (ls_stats['instructions']) / ls_stats['cpu-cycles'] if ls_stats['cpu-cycles'] != 0 else 0
            state[4] = state[4]/configuration[ls_image]['ipc']

            # 6. #cntrs
            state[5] = self.min_max(cnts, 1, 7)

        else:
            state = [0, 0, 0, 0, 0, 0]

        # 6. Priority
        norm_prio = 0 if(self.data[app]['priority'] == -1) else (self.min_max(self.data[app]['priority'],1,80))

        # 7. Affinity
        norm_affin = 0 if(self.data[app]['affin'] == -1) else (self.min_max(self.data[app]['affin'],0,16))

        # 8. P_low
        # 9. P_high
        # 10. A_other
        p_low, p_high, a_other = self.get_others(app)
        norm_p_low = (p_low - 0) / 10
        norm_p_high = (p_high - 0) / 10
        norm_a_other = (a_other - 0) / 16

        # 11. Employ futex lock; It is a boolean variable that denotes if an application uses futex lock or not. In our case we know that OD only uses the futex lock.
        lock_var = 0
        if(app == "od"):
            lock_var = 1

        # 12. Resource fairness: self.cur_spi_degrad

        # 2.9GHz is the CPU frequency of the machine
        S_spi = state[0] + 1/(2900000000*ls_stats["IPC"])
        S_spi_iso = (configuration[ls_image]['min_cpu_wait']/configuration[ls_image]['instruction']) + 1/(2900000000*configuration[ls_image]['ipc'])

        S_fair = self.cur_spi_degrad
        S_perf = S_spi_iso/S_spi

        # print(app, state)

        # 0.5 is the minimum performance degradation that an application can sustain!!
        # Note that we have only 1 application in this code, so we are checking S_perf directly.
        # In case you have multiple functions, you need to ensure that S_perf >= 0.5 for all the LS applications.
        if((S_fair > self.tau) and (S_perf >= 0.5)):
            self.data[app]['reward'] =  self.sfair_weight*S_fair + self.sperf_weight*S_perf
        else:
            self.data[app]['reward'] = -1

        self.data[app]['state'] = state + [norm_prio, norm_affin, norm_p_low, norm_p_high, norm_a_other, lock_var, self.cur_spi_degrad]

        self.data[app]['prev_cpu_state']['cpu_wait_time'] = self.data[app]['cur_cpu_state']['cpu_wait_time']
        self.data[app]['prev_cpu_state']['nvcs'] = self.data[app]['cur_cpu_state']['nvcs']

    def get_perf_gain(self, num_reqs, stat, app):
        iso_gain = {"dummy": 1.8613898504505884e-06}
        gain = 0
        if(num_reqs != 0):
            if(stat['cpu_wait_time'] > self.data[app]['prev_cpu_state']['cpu_wait_time']):
                gain += (stat['cpu_wait_time'] - self.data[app]['prev_cpu_state']['cpu_wait_time'])/stat['instructions']
            else:
                gain += stat['cpu_wait_time']/stat['instructions']
            
            if(app in ["pg", "dv", "ra", "vp", "ir"]):
                self.data[app]['prev_cpu_state']['cpu_wait_time'] = stat['cpu_wait_time']

            gain += 1 / (2400000000*stat['IPC'])
            print("Perf Gain", app, gain)
            return iso_gain[app]/gain
        
        else:
            return 0

    def train(self):
        # Wait until the epoch ends
        while(os.path.isfile("/tmp/serverless_lock")):
            time.sleep(1)
        
        # Signal the framework that we will be providing the action
        os.system("touch /tmp/learning")

        timestamp, running_apps = self.get_ld_info()
        dummy_num_reqs = self.get_exec_details(int(timestamp))

        dummy_stats = self.get_stats()

        spis = []
        spis.append(self.get_perf_gain(dummy_num_reqs, dummy_stats, "dummy"))

        non_zero_spi = []
        for spi_values in spis:
            if(spi_values != 0):
                non_zero_spi.append(spi_values)

        print(non_zero_spi, min(non_zero_spi), max(non_zero_spi))

        if(len(non_zero_spi) == 0):
            return  #Nothing is running on the system
        else:
            # Degradation in the IPC.
            if(max(non_zero_spi) != 0):
                self.cur_spi_degrad = min(non_zero_spi) / max(non_zero_spi)
            else:
                return # Something went wrong

        for app in self.app:
            if app not in running_apps.keys():
                continue

            if(app == "dummy"):
                self.get_state(app, dummy_stats, "faasched:dummy", dummy_num_reqs, running_apps["dummy"], running_apps)

        # End of the state collection.


        if(not self.init_state):
            for app in self.app:
                if app not in running_apps.keys():
                    continue
                self.train_policy(self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['reward'], self.data[app]['value'], self.data[app]['state'])
        else:
            self.init_state = False


        # Take an action.
        self.last_cpu = 0
        total_affin = 0

        for app in self.app:
            if app not in running_apps.keys():
                self.data[app]['priority'], self.data[app]['affin'], self.data[app]['cpus'] = 0, -1, -1
                continue

            if (self.data[app]['priority'] == 0) and (self.data[app]['affin'] == -1) and (self.data[app]['cpus'] == -1):
                if(app == "od"):
                    self.data[app]['priority'] = 50
                    self.data[app]['affin'] = -1 
                    self.data[app]['cpus'] = -1
                else:
                    self.data[app]['priority'] = 50
                    self.data[app]['affin'] = 3
                    self.data[app]['cpus'] = -1

            while(True):
                self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['value'] = self.get_policy(self.data[app]['state'], app)
                print(self.data[app]['action'], self.data[app]['priority'], self.data[app]['affin'])

                if(app == "od"):
                    if(self.data[app]['priority'] >= 80) and (self.data[app]['action'] in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
                        self.data[app]['reward'] = -1
                        self.train_policy(self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['reward'], self.data[app]['value'], self.data[app]['state'])

                    elif(self.data[app]['priority'] <= 1) and (self.data[app]['action'] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
                        self.data[app]['reward'] = -1
                        self.train_policy(self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['reward'], self.data[app]['value'], self.data[app]['state'])

                    elif(self.data[app]['action'] in [2, 3, 4, 5, 6, 7, 8, 9, 10]):
                        self.data[app]['reward'] = -1
                        self.train_policy(self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['reward'], self.data[app]['value'], self.data[app]['state'])

                    else:
                        break

                else:
                    if(self.data[app]['priority'] <= 1):
                        if((self.data[app]['affin'] >= 16) and (self.data[app]['action'] in [1, 3, 4, 7, 5, 8])):
                            self.data[app]['reward'] = -1
                            self.train_policy(self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['reward'], self.data[app]['value'], self.data[app]['state'])
                        elif((self.data[app]['affin'] == -1) and (self.data[app]['action'] in [1, 3, 4, 7, 9, 2])):
                            self.data[app]['reward'] = -1
                            self.train_policy(self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['reward'], self.data[app]['value'], self.data[app]['state'])
                        elif(self.data[app]['action'] in [1, 3, 4, 7]):
                            self.data[app]['reward'] = -1
                            self.train_policy(self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['reward'], self.data[app]['value'], self.data[app]['state'])
                        else:
                            total_affin += self.data[app]['affin'] if self.data[app]['affin']!=-1 else 0
                            break
                    
                    elif(self.data[app]['priority'] >= 80):
                        if((self.data[app]['affin'] >= 16) and (self.data[app]['action'] in [0, 2, 5, 6, 3, 8])):
                            self.data[app]['reward'] = -1
                            self.train_policy(self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['reward'], self.data[app]['value'], self.data[app]['state'])
                        elif((self.data[app]['affin'] == -1) and (self.data[app]['action'] in [0, 2, 5, 6, 9, 4])):
                            self.data[app]['reward'] = -1
                            self.train_policy(self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['reward'], self.data[app]['value'], self.data[app]['state'])
                        elif(self.data[app]['action'] in [0,2,5,6]):
                            self.data[app]['reward'] = -1
                            self.train_policy(self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['reward'], self.data[app]['value'], self.data[app]['state'])
                        else:
                            total_affin += self.data[app]['affin'] if self.data[app]['affin']!=-1 else 0
                            break
                    
                    elif((self.data[app]['affin'] >= 16) and (self.data[app]['action'] in [3,5,8])):
                        self.data[app]['reward'] = -1
                        self.train_policy(self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['reward'], self.data[app]['value'], self.data[app]['state'])

                    elif((self.data[app]['affin'] == -1) and (self.data[app]['action'] in [2, 4, 9])):
                        self.data[app]['reward'] = -1
                        self.train_policy(self.data[app]['policy_dist'], self.data[app]['action'], self.data[app]['reward'], self.data[app]['value'], self.data[app]['state'])

                    else:
                        total_affin += self.data[app]['affin'] if self.data[app]['affin']!=-1 else 0
                        break
        
        # Since, we have only one application the cores are always allocated in the increasing order of the CORE IDs.
        for app in self.app:
            if app not in running_apps:
                continue

            self.data[app]['priority'], self.data[app]['affin'] = self.take_action(self.data[app]['action'], self.data[app]['priority'], self.data[app]['affin'])

            if (self.data[app]['affin'] != -1) and (app != "od"):
                self.data[app]['cpus'] = ",".join(cpu_layout[self.last_cpu:self.last_cpu+self.data[app]['affin']])
                self.last_cpu += self.data[app]['affin']
            else:
                self.data[app]['cpus'] = -1

        affin_rs = ",".join(cpu_layout[self.last_cpu:])

        os.system("{} {} {} {} {} {} {} {} {} {} {} {}".format(self.adjust_prio_command, self.data['bs']['priority'], self.data['bs']['cpus'], self.data['mr']['priority'], self.data['mr']['cpus'], self.data['od']['priority'], self.data['od']['cpus'], self.data['sa']['priority'], self.data['sa']['cpus'], self.data['eg']['priority'], {self.data['eg']['cpus']}, affin_rs))

        time.sleep(30) # Let the docker update it's pid and stuffs

    def handler(self):
        while(True):
            if(os.path.isfile("/tmp/serverless_end")): # This flag denotes that the experiment is completed
                break
            elif(os.path.isfile("/tmp/serverless_lock")): # The function has started execution
                obj.save()
                self.train()
                os.system("rm -rf /tmp/learning") # Signal the framework that the actions are taken

        print(f"Model Saved")

# For inferencing, set epsilon to 0, and comment the training part.
obj = ServerlessAgent()
obj.handler()