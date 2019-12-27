import os
import pickle
import multiprocessing as mp

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

import pdb

class ParallelSim(object):
    def __init__(self, processes):
        self.pool = mp.Pool(processes=processes)
        self.total_processes = 0
        self.completed_processes = 0
        self.results = []

    def add(self, func, args):
        print(args)
        self.pool.apply_async(func=func, args=args, callback=self.complete)
        self.total_processes += 1

    def complete(self, results):
        self.results.append(results)
        self.completed_processes += 1
        print('-- processed {:d}/{:d}'.format(self.completed_processes,
                                              self.total_processes), end='\r')

    def run(self):
        self.pool.close()
        self.pool.join()

    def get_results(self):
        return self.results

class ArgoverseDataset(Dataset):
    def __init__(self, data_dir, map_version, testset=False, num_workers=None, cache_file=None):
        """
        Args:
        :param data : List of [scene_id, scene_image, number_agents, past_list, future_list,
                               encode_coordinates, decode_coordinates]
        """
        if map_version=='1.3':
            self.map_version = map_version
        else:
            raise("Invalid map: only 1.3 is valid")
        self.testset = testset
        self.img_transform = transforms.Compose([transforms.Resize((64, 64)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.5, 0.5, 0.5],
                                                                      [0.2, 0.2, 0.2])]
                                                )

        self.num_workers = num_workers
      
        # Extract Data:
        if cache_file is None:
            self.get_data(data_dir)
        else:
            if os.path.isfile(cache_file):
                self.load_cache(cache_file)
            else:
                self.get_data(data_dir, save_cache_dir=cache_file)

    def __getitem__(self, idx):
        # Create one past list and future list with all the
        past_agents_traj = self.past_agents_traj_list[idx]
        past_agents_traj_len = self.past_agents_traj_len_list[idx]

        decode_start_vel = self.decode_start_vel[idx]
        decode_start_pos = self.decode_start_pos[idx]
        decode_start_pos_city = self.decode_start_pos_city[idx]
        
        scene_id = self.scene_id[idx]

        map_image = Image.open(self.scene_map_paths[idx])
        map_image = self.img_transform(map_image)

        if self.testset:
          episode = (past_agents_traj, past_agents_traj_len, decode_start_vel, decode_start_pos, decode_start_pos_city, map_image, scene_id)
        else:
          future_agents_traj = self.future_agents_traj_list[idx]
          future_agents_traj_len = self.future_agents_traj_len_list[idx]
          episode = (past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, decode_start_vel, decode_start_pos, decode_start_pos_city, map_image, scene_id)

        return episode

    def __len__(self):
        return len(self.scene_id)

    def load_cache(self, cache_dir):
        with open(cache_dir, 'rb') as f:
            results = pickle.load(f)

        self.past_agents_traj_list, self.past_agents_traj_len_list, self.future_agents_traj_list,\
        self.future_agents_traj_len_list, self.decode_start_vel, self.decode_start_pos,\
        self.decode_start_pos_city, self.scene_map_paths, self.scene_id = list(zip(*results))

    def get_data(self, root_dir, save_cache_dir=None):
        print(f'Extracting data from: {root_dir}')

        sub_directories = os.listdir(root_dir)
        sub_directories.sort()
        path_lists = []

        for i, sub_directory in enumerate(sub_directories):
            print(f'{sub_directory}')
            sub_directory = root_dir + sub_directory + '/'
            path_lists.extend(self.extract_directory(sub_directory))

        # if self.num_workers:
        #     num_processes = self.num_workers
        # else:
        #     num_processes = mp.cpu_count()

        # runner = ParallelSim(processes=num_processes)

        results = []
        for path_list in path_lists:
            results.append(self.extract_trajectory_multicore(path_list))
            # runner.add(self.extract_trajectory_multicore, (path_list, ))

        # runner.run()
        # results = runner.get_results()

        if save_cache_dir is not None:
            with open(save_cache_dir, 'wb') as f:
                pickle.dump(results, f) 

        self.past_agents_traj_list, self.past_agents_traj_len_list, self.future_agents_traj_list,\
        self.future_agents_traj_len_list, self.decode_start_vel, self.decode_start_pos,\
        self.decode_start_pos_city, self.scene_map_paths, self.scene_id = list(zip(*results))

        print('Extraction Compltete!\n')

    def extract_directory(self, directory):

        scene_segments = os.listdir(directory)
        scene_segments.sort(key=lambda x: int(x[-8:], 16))
        path_lists = []

        num_segments = len(scene_segments)
        for i, scene_segment in enumerate(scene_segments):
            observation_dir = directory + scene_segment + '/observation'
            reference_frames = os.listdir(observation_dir)
            reference_frames.sort()
            
            for j in range(len(reference_frames)):
                path_lists.append((directory, scene_segment, reference_frames[j]))
            
            print('{:d}/{:d}'.format(i, num_segments), end='\r')

        return path_lists

    def extract_trajectory_multicore(self, path_list):
        print(path_list, end='\r')
        directory, scene_segment, reference_frame = path_list
        observation_path = directory + scene_segment + '/observation/' + reference_frame
        map_path = directory + scene_segment + '/map/v{:s}/'.format(self.map_version) + reference_frame.replace('pkl', 'png')

        with open(observation_path, 'rb') as f:
            observation_df = pickle.load(f)

        prediction_df = None
        if not self.testset:
          prediction_path = directory + scene_segment + '/prediction/' + reference_frame
          with open(prediction_path, 'rb') as f:
              prediction_df = pickle.load(f)

        past_agent_traj, past_agent_traj_len, future_agent_traj, future_agent_traj_len, decode_start_vel, decode_start_pos, decode_start_pos_city = self.extract_trajectory_info(observation_df, prediction_df)
        scene_id = (scene_segment + '/' + reference_frame)

        return (past_agent_traj, past_agent_traj_len, future_agent_traj, future_agent_traj_len, decode_start_vel, decode_start_pos, decode_start_pos_city, map_path, scene_id)


    def extract_trajectory_info(self, obv_df, pred_df):
        """
        Extracts the past and future trajectories of the agents as well as the encode and decode
        coordinates.
        """
        agent_obv_df = obv_df[obv_df['class'] == 'AGENT']
        past_agent_traj = agent_obv_df[['X', 'Y']].to_numpy().astype(np.float32)
        decode_start_vel = past_agent_traj[-1] - past_agent_traj[-2]
        decode_start_pos = past_agent_traj[-1]
        decode_start_pos_city = agent_obv_df[['X_city', 'Y_city']].iloc[-1].to_numpy()

        past_agent_traj_len = past_agent_traj.shape[0]

        future_agent_traj = None
        future_agent_traj_len = None
        if pred_df is not None:
          agent_pred_df = pred_df[pred_df['class'] == 'AGENT']
          future_agent_traj = agent_pred_df[['X', 'Y']].to_numpy().astype(np.float32)
          future_agent_traj_len = future_agent_traj.shape[0]

        return past_agent_traj, past_agent_traj_len, future_agent_traj, future_agent_traj_len, decode_start_vel, decode_start_pos, decode_start_pos_city
