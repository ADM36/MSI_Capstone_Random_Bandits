"""
The purpose of this file is to stub out functionality in the form of functions we'll likely need.
As of now, everything here is a draft of an idea and can/should be edited as we see fit.
"""
import pandas as pd
import sys, os
import csv
from datetime import datetime
import json
import argparse # for command-line arguments.
import ast # for use of the 'literal' function which can be used to parse arguments nicely.
import numpy as np # for easy averages and math
from collections import Counter

class Bandits(object):
    """
        This class will determine which query to run by way of the UCB
        multi-armed bandit algorithm, as implemented by Li et al. 2016.
        Methods included herein will also write the results of any update
        to the UCB of each query to a file. If such a file already exists,
        it will pick up from where it last left off with the new data.
    """

    def __init__(self, run_id='test', continuation=False, confidence_modifier=1.0):
        """
        Initializes an instance of <Bandits>.

        Parameters:
            - run_id (str): A string ID for this version of <Bandits>. Use the same run_id
                    as long as you are working with the same experiment to ensure that
                    you are accessing the correct model (this run_id sets the output
                    filepaths).
            - continuation (bool): Determines if this instance will continue a model
                    experiment that is already being performed or if it will create a new
                    model. True to continue an experiment with <run_id>, False to start
                    a new one.
        """

        self.confidence_modifier = confidence_modifier
        self.run_id = run_id
        if continuation:
            # Continuing from an existing file storing the ucb information
            self.ucb_lookup = self.read_ucb()
        else:
            # Initialize a new bandit chain.
            self.ucb_lookup = {"total_times_played":0, "levers":{}}


    def __str__(self,):
        """
        This returns a simplified summary of what the state of each query's UCB is
        currently at.
        """

        UCBs = [f"{k}: {v['ucbs'][-1]}" for k,v in self.ucb_lookup['levers'].items()]
        description = "\n\n\nLever: Current UCB\n\n" + '\n'.join(UCBs)

        return description


    def initialize_lever(self,new_lever_dict):
        """
        Initialize the a lever by providing its first score. Update self.ucb_lookup
        for each lever. Provide option to initialize just one lever or many.

        Parameters:
            - new_lever_dict (dict): A dictionary that contains keys of the new levers
                    and values as the first score (reward).

        Returns:
            - None
        """

        for k,v in new_lever_dict.items():
            if k in self.ucb_lookup['levers']:
                raise SystemExit(f'Error:\n{k} already exists as a lever! It was not added.')
            self.ucb_lookup['levers'][k] = {}
            self.ucb_lookup['levers'][k]['scores'] = [v]
            self.ucb_lookup['levers'][k]['ucbs'] = []
            self.ucb_lookup['levers'][k]['percent_pulls'] = []
            # Note: The ucb and percent_pulls will be updated when the algorithm runs.
            self.ucb_lookup['total_times_played'] += 1


    def read_ucb(self,):
        """
        Imports a ucb score file that has been written by a previous run of this bandit
        algo.

        Files should be structured as:

            {
                "total_times_played" : <integer number of total times played>
                "levers" : {
                    <lever1>:{'scores':<score list for lever 1>,
                              'ucbs':<list of ucbs for lever 1>,
                              'percent_pulls':<list of percent of total pulls for this lever>}
                    <lever2>:{'scores':<score list for lever 2>,'ucbs':<list of ucbs for lever 2>,
                    ...etc
                }
            }

        where each score is the "reward" from each pull of the lever, and the list of ucbs is
        the UCB calculated after each round of "rewards"
        """

        #! Need to reimplement lines below, throwing an error
        # try self.ucb_lookup:
        #     raise SystemExit("Error:\nYou cannot overwrite a UCB lookup table by importing a new one.")
        # except NameError:
        #     pass

        with open(f"ucb_scores/{self.run_id}_ucb_scores.json", 'r') as f:
            ucb_lookup = json.loads(f.read())

        return ucb_lookup


    def write_ucb(self,):
        """
        Writes out the current UCB scores to a _ucb_scores file and also
        appends to a log file which keeps track of the ucb scores for each
        iteration of this model.

        TODO: Make the log file that keeps track of all runs and their individual
        scores.
        """

        with open(f"ucb_scores/{self.run_id}_ucb_scores.json", 'w') as f:
            f.write(json.dumps(self.ucb_lookup))


    def select_lever(self,):
        """
        Returns the "lever" (i.e. query) with the highest UCB that should be queried next.
        """

        if len(self.ucb_lookup.keys() == self.num_arms):
            return max(self.ucb_lookup, key=self.ucb_lookup.get)
        else:
            raise SystemExit("Error:\nNumber of UCB keys does not match number of arms. Please initialize new arm.")


    def calculate_new_ucb(self, lever):
        """
        Calculates the new UCB given the data present in self.ucb_lookup.

        Parameters:
        - lever (str): A dictionary key for ucb_lookup that determines which lever
            to calculate the UCB for.

        Returns:
        - (float): A float value with which to update the UCB for this lever in
            ucb_lookup.
        """

        lever_data = self.ucb_lookup['levers'][lever]
        Ni = float(len(lever_data['scores'])) # Number of pulls of this lever
        t = self.ucb_lookup['total_times_played'] # Number of all total pulls

        rewards = np.array(lever_data['scores'])
        mean_reward = rewards.mean()
        confidence_interval = self.confidence_modifier * np.sqrt(np.log(t) / Ni)

        percent_pull = Ni/t*100.0

        return mean_reward + confidence_interval, percent_pull


    def determine_next_lever(self):
        """
        Retrieves the most recent UCBs and returns the lever with the highest.
        """

        highest_UCB = -999
        next_lever = None
        for lever,data in self.ucb_lookup['levers'].items():
            last = data['ucbs'][-1]
            if last > highest_UCB:
                highest_UCB = last
                next_lever = lever

        return next_lever


    def run_algorithm_once(self,lever,score):
        """
        TODO: Check if self.ucb_lookups exists, if it does, calculate new UCBs and perform
        select_lever. This function will take in the most recent reward data, then calc
        new UCBs for each lever, then return the lever to pull next.

        1. Take in lever name and new score
        2. Update UCBS and lookup table
        3. Return new lever.
        """

        if lever not in self.ucb_lookup['levers']:
            ans = 'q'
            while ans not in ['y','n']:
                ans = input(f"{lever} doesn't exist! Initialize it first? (y/n): ")
                if ans == 'y':
                    print(f'\nAdding lever {lever}...')
                    self.initialize_lever({lever: score})
                elif ans == 'n':
                    raise SystemExit('\nLever not added. Exiting...')
        else:
            self.ucb_lookup['levers'][lever]['scores'].append(score)

        # Update total pulls:
        self.ucb_lookup['total_times_played'] += 1

        # Update UCBs:
        for lev in self.ucb_lookup['levers']:
            ucb, pp = self.calculate_new_ucb(lev)
            self.ucb_lookup['levers'][lev]['ucbs'].append(ucb)
            self.ucb_lookup['levers'][lev]['percent_pulls'].append(pp)

        # Return the next lever.
        return self.determine_next_lever()

def calculate_kappa(cleaned_df):
    """
    Process select results from Turkers and calculate their Fleiss' Kappa.

    Args:
        cleaned_df (DataFrame): pandas DF of processed mTurk results.

    Returns:
        fleiss_kappa (float): Fleiss' Kappa calculated for results.
    """
    

def process_mturk_results(file_path, gold_samples):
    """
    Read in the results from a round of mTurk labeling.

    Args:
        file_path (str): path to a mTurk results CSV
        gold_samples (dict): keys are sample ids and values are expected bool

    Returns:
        a pandas dataframe with rows filtered for certain criteria

    """
    infile_df = pd.read_csv(file_path)
    print(f"Initial mTurk results df shape: {infile_df.shape}")
    print(f"Num of unique workers: {len(infile_df.WorkerId.unique())}")

    # filter out HITs where WorkTimeInSeconds < # seconds
    MIN_WORK_SECONDS = 30
    under_min_work_df = infile_df[infile_df["WorkTimeInSeconds"]<MIN_WORK_SECONDS]
    under_min_counter = Counter(under_min_work_df.WorkerId)
    all_worker_counter = Counter(infile_df.WorkerId)

    PERCT_THRESHOLD = 0.3
    ids_to_remove = []
    for k,v in under_min_counter.items():
        total_val = all_worker_counter[k]
        if v/total_val > PERCT_THRESHOLD:
            ids_to_remove.append(k)
    print(f"Num workers to remove for MIN_WORK filter: {len(ids_to_remove)}")
    
    infile_df = infile_df[~infile_df.WorkerId.isin(ids_to_remove)]
    print(f"mTurk results df after MIN_WORK filter: {infile_df.shape}")

    print(f"Num of gold samples: {len(gold_samples.keys())}")

    # filter out HITs that failed gold samples
    # gold samples are key, value pairs of 'HITId', 'Answer.yes.1'
    gold_samples_df = infile_df[infile_df['HITId'].isin(list(gold_samples.keys()))]

    # bad_assignment_ids = [] #* no longer needed
    bad_worker_ids = []

    for index, row in gold_samples_df.iterrows():
        if gold_samples[row['HITId']] == row['Answer.yes.1']:
            continue
        else:
            # bad_assignment_ids.append(row['AssignmentId']) #* no longer needed
            bad_worker_ids.append(row['WorkerId'])

    # infile_df = infile_df[~infile_df.AssignmentId.isin(bad_assignment_ids)] #! removes row for a bad answer even if all rows for a worker aren't removed in filtering below

    # filter out Workers who failed more than 2 gold samples
    dictOfElems = dict(Counter(bad_worker_ids))

    workers_to_delete = []

    for k, v in dictOfElems.items():
        if dictOfElems[k] > 2:
            workers_to_delete.append(k)
    print(f"Workers to delete: {workers_to_delete}")

    infile_df = infile_df[~infile_df.WorkerId.isin(workers_to_delete)]
    print(f"mTurk results df after bad_worker filter: {infile_df.shape}")

    # # filter out those where Answer.no.0 = true but Answer.Directive Sentence is either < 2 or >= len of Input.TEXT
    # infile_df = infile_df[((infile_df["Answer.no.0"]==True)&(len(infile_df["Answer.Directive Sentence"])<2))|(len(infile_df["Answer.Directive Sentence"])>=len(infile_df['Input.TEXT']))]
    # print(infile_df.shape)

    outfile_df = infile_df

    return outfile_df

if __name__ == "__main__":

    # test_path = "Batch_3949723_batch_results.csv"
    # gold_samples = {4:True,11:True,28:False,29:False,49:True}

    # test_path = "fake_MTurk_results.csv"
    test_path = "fake_MTurk_results_2020-3-29-2142.csv"
    gold_samples = {1: False, 11:True, 38:True, 40:True, 97:False}

    cleaned_file = process_mturk_results(test_path, gold_samples)
    
