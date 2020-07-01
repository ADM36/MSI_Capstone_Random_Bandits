import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import uniform, norm, gamma, binom
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

    def __init__(self, run_id='test', continuation=False, confidence_modifier=0.2):
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
        TODO: Docstring
        """

        if lever not in self.ucb_lookup['levers']:
            ans = 'q'
            while ans not in ['y','n']:
                # ans = input(f"\nLever '{lever}' doesn't exist! Initialize it first? (y/n): ")
                ans='y'
                if ans == 'y':
                    print(f"Adding lever '{lever}'...")
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


def multiple_MTurk_accuracy(rate, ground_truth, num_workers):

    '''
    Input: accuracy rate desired for MTurker, ground truth labels and the number
    of MTurk workers

    Description: This function creates dummy response labels for n MTurk workers,
    where n = num_workers. Response labels for each MTurk worker is determined by
    the accuracy rate (rate) provided by the user. For each ground truth label in
    ground_truth, the workers response is determined at random, but with a probability
    equal to the accuracy rate provided.

    Returns: A list of lists. Each list is a single Murk workers' response labels against
    the ground truth labels provided. The number of lists is equal to num_workers.
    '''

    all_MTurker_labels = []

    for i in range(num_workers):

        MTurker_labels = []
        if rate == 1.0:
            MTurker_labels = ground_truth
        else:
            for gt_label in ground_truth:
                if gt_label == 0:
                    MTurker_labels.append(np.random.choice([0, 1], p=[rate, 1.0 - rate]))
                else:
                    MTurker_labels.append(np.random.choice([0, 1], p=[1.0 - rate, rate]))
        all_MTurker_labels.append(MTurker_labels)

    return all_MTurker_labels



def single_MTurk_accuracy(rate, ground_truth):

    '''
    Input: accuracy rate desired for MTurker, ground truth labels

    Description: This function creates dummy response labels for a single MTurk worker.
    Response labels for the worker is determined by the accuracy rate (rate) provided
    by the user. For each ground truth label in ground_truth, the worker's response is
    determined at random, but with a probability equal to the accuracy rate provided.

    Returns: A list of a single Murk workers response labels against
    the ground truth labels provided.
    '''

    MTurker_labels = []
    if rate == 1.0:
        return ground_truth
    else:
        for gt_label in ground_truth:
            if gt_label == 0:
                MTurker_labels.append(np.random.choice([0, 1], p=[rate, 1.0 - rate]))
            else:
                MTurker_labels.append(np.random.choice([0, 1], p=[1.0 - rate, rate]))

    return MTurker_labels


def different_rate_Mturkers(list_of_rates, ground_truth):

    '''
    Input: list of accuracy rates desired for MTurkers, ground truth labels

    Description: This function creates dummy response labels for n MTurk workers,
    where n = len(list_of_rates). Response labels for each MTurk worker is determined by
    the accuracy rate (list_of_rates[i], where i represents a single MTurker) provided
    by the user. For each ground truth label in ground_truth, the workers response
    is determined at random, but with a probability equal to the accuracy rate
    (list_of_rates[i]) provided.

    Returns: A list of lists. Each list is a single Murk workers' response labels against
    the ground truth labels provided. The number of lists is equal to len(list_of_rates).
    '''

    all_worker_results = []

    for rate in list_of_rates:

        single_worker_results = single_MTurk_accuracy(rate, ground_truth)

        all_worker_results.append(single_worker_results)

    return all_worker_results


def fake_MTurk_results_file_generator(Mturkers_data):

    '''
    Input: A list of lists. Each list is a single Murk workers' response labels against
    the ground truth labels provided. The number of lists is equal to the number of workers.

    Description: This function creates a pandadas dataframe and CSV file containing
    Mturker label results.

    Returns: A pandas dataframe and a CSV file. The columns are:
        - WorkerId: Unique id for each worker.
        - HitId: The unique HIT for each label (text data where directive is being
        identified as existing or not).
        - Answer.no.0: Column indicating if labeler said there was NO directive. 1 means
        they marked "no directive", O means otherwise.
        - Answer.yes.1: Column indicating if labeler said there IS a directive present. 1
        means they marked "yes, directive", 0 means otherwise.
    '''

    worker_id = 0

    # each AssignmentId needs to be unique
    AssignmentId = 0

    all_rows = []

    for worker in Mturkers_data:

        worker_id += 1

        hit_id = 0

        for label in worker:

            ### START VARS ###
            AssignmentId += 1

            hit_id += 1

            HITTypeId = "dummy text"
            Title = "dummy text"
            Description = "dummy text"
            Keywords = "dummy text"
            Reward = "dummy text"
            CreationTime = "dummy text"
            MaxAssignments = 99
            RequesterAnnotation = "dummy text"
            AssignmentDurationInSeconds = 99
            AutoApprovalDelayInSeconds = 99
            Expiration = "dummy text"
            NumberOfSimilarHITs = None
            LifetimeInSeconds = None
            #AssignmentId goes here but already create above
            #WorkerId goes here but already create above
            AssignmentStatus = "dummy text"
            AcceptTime = "dummy text"
            SubmitTime = "dummy text"
            AutoApprovalTime = "dummy text"
            ApprovalTime = None
            RejectionTime = None
            RequesterFeedback = None
            WorkTimeInSeconds = np.random.randint(low = 1, high = 100) # generate rand val between 1 and 100
            LifetimeApprovalRate = "dummy text"
            Last30DaysApprovalRate = "dummy text"
            Last7DaysApprovalRate = "dummy text"
            Input_TEXT = "dummy text"
            Answer_Directive_Sentence = "dummy text"
            Answer_no_0 = 0 if label == 1 else 1
            Answer_yes_1 = 1 if label == 1 else 0
            Approve = None
            Reject = None
            ### END VARS ###

            new_row = [hit_id, HITTypeId, Title, Description, Keywords, \
                Reward, CreationTime, MaxAssignments, RequesterAnnotation,\
                    AssignmentDurationInSeconds, AutoApprovalDelayInSeconds,\
                        Expiration, NumberOfSimilarHITs, LifetimeInSeconds,\
                            AssignmentId, worker_id, AssignmentStatus, AcceptTime,\
                                SubmitTime, AutoApprovalTime, ApprovalTime,\
                                    RejectionTime, RequesterFeedback, WorkTimeInSeconds,\
                                        LifetimeApprovalRate, Last30DaysApprovalRate, Last7DaysApprovalRate,\
                                            Input_TEXT, Answer_Directive_Sentence, Answer_no_0,\
                                                Answer_yes_1, Approve, Reject]

            all_rows.append(new_row)

    fake_MTurk_result_file = pd.DataFrame(all_rows,
    columns=['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward',
       'CreationTime', 'MaxAssignments', 'RequesterAnnotation',
       'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds',
       'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds',
       'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime',
       'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime',
       'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate',
       'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Input.TEXT',
       'Answer.Directive Sentence', 'Answer.no.0', 'Answer.yes.1', 'Approve',
       'Reject'])

    #time_stamp = f"{datetime.datetime.now().year}-{datetime.datetime.now().month}-{datetime.datetime.now().day}-{datetime.datetime.now().hour}{datetime.datetime.now().minute}"

    #fake_MTurk_result_file.to_csv(f"fake_MTurk_results_{time_stamp}.csv", index = False)

    return fake_MTurk_result_file


class SyntheticLever(object):

    def __init__(self, name, mean_num_directives, extend, total_samples = 100):

        self.name = name
        self.mean_num_directives = mean_num_directives
        self.extend = extend
        self.total_samples = total_samples

    def __str__(self):

        return f"Lever '{self.name}' has a mean fraction of directives: {self.mean_num_directives} +/- {self.extend}"

    def generate_ground_truth(self):
        p = np.random.uniform(self.mean_num_directives-self.extend, self.mean_num_directives+self.extend)
        return binom.rvs(n=1,p=p,size=self.total_samples)


def determine_reward(mturk_df, threshold = 5):
    ids = mturk_df['HITId']
    labels = mturk_df['Answer.yes.1']

    label_dict = {}
    for i,_ in enumerate(ids):
        if ids[i] not in label_dict:
            label_dict[ids[i]] = [labels[i]]
        else:
            label_dict[ids[i]].append(labels[i])

    # calculate the number of confident samples ( > threshold agreed it was a positive sample)
    number_confident = sum([1 if sum(v) >= threshold else 0 for k,v in label_dict.items()])
    number_identified = sum([1 if sum(v) >= 1 else 0 for k,v in label_dict.items()])
    #print(number_confident)

    # Calculating the percentage of labelers that labeled a 1 for each document.
    reward_dict = {k: (np.array(v).sum()/len(v)) for k,v in label_dict.items()}
    # reward_dict = {k: (np.array(v).sum()/len(v)) if np.array(v).sum() > 3 else 0 for k,v in label_dict.items()}

    # Return the sum of the reward for all 100 documents, then divide by 100.
    return np.array(list(reward_dict.values())).sum()/100.0, number_confident, number_identified


class SimulationPipeline(object):


    def __init__(self, levers={'a':0.5,'b':0.6,'c':0.7,'d':0.8},labelers=[0.95, 0.95, 0.95, 0.95, 0.95, 0.95]):

        self.levers = levers
        self.labelers = labelers


    def run_simulation(self, iterations=1000):

        # Get instance variables.
        labelers = self.labelers
        levers = self.levers

        # Create total reward list. This has the total number of documents that
        # above some threshold labeled as positive.
        total_reward = []

        # Create the total labeled list. This has the total number of documents that
        # at least received a single positive label.
        total_labeled = []


        # Create Bandits object.
        bandits = Bandits(run_id='test_pipeline',confidence_modifier=0.2)

        # Initialize levers:
        # First, make the synthetic generators.
        lev = {k: SyntheticLever(k, v, 0.05) for k,v in levers.items()}
        # Then perform initial query for each lever and get MTurk labels.
        for k,lever in lev.items():
            # generate ground truth
            gt = lever.generate_ground_truth()
            # generate labels from labelers
            labels = different_rate_Mturkers(labelers, gt)
            # generate df from MTurk results
            mturk_df = fake_MTurk_results_file_generator(labels)
            # calculate reward
            reward, nc, ni = determine_reward(mturk_df)
            # initialize lever with bandits
            next_lever = bandits.run_algorithm_once(lever.name,reward)

        print(f"\n\nPerforming simulation for {iterations} iterations.\n\tLevers: {self.levers}\n\tScheduled labeler accuracy: {self.labelers}\n")

        for i in range(iterations):

            if i%100 == 0:
                print(f"\t...iteration: {i}/{iterations}")
            gt = lev[next_lever].generate_ground_truth()
            labels = different_rate_Mturkers(labelers,gt)
            mturk_df = fake_MTurk_results_file_generator(labels)
            reward, nc, ni = determine_reward(mturk_df)
            #print(reward)
            if len(total_reward) == 0:
                total_reward.append(nc)
            else:
                total_reward.append(total_reward[-1]+nc)

            if len(total_labeled) == 0:
                total_labeled.append(ni)
            else:
                total_labeled.append(total_labeled[-1]+ni)
            next_lever = bandits.run_algorithm_once(lev[next_lever].name, reward)

        self.bandits = bandits
        self.total_reward = total_reward
        self.total_labeled = total_labeled



    def plot_bandits(self, savefile='', labelers_mark=True, plot_reward=True):

        x = self.bandits

        to_plot = {}
        for lever, lever_data in x.ucb_lookup['levers'].items():
            to_plot[lever] = lever_data['percent_pulls']

        fig = plt.figure(figsize=[12,8])
        ax = fig.add_subplot(111)
        for lever, plot_data in to_plot.items():
            ax.plot(range(len(plot_data)), plot_data, label=f"{lever} (mean reward = {self.levers[lever]})")

        # if plot_reward:
        #     ax2 = ax.twinx()
        #     ax2.plot(range(len(self.total_reward)), self.total_reward, lw=3, alpha=0.5, color='grey')

        if labelers_mark:
            ax.text(0.1, 0.02, f"Labeler Accuracies: {', '.join([str(x) for x in self.labelers])}", transform=ax.transAxes)
        ax.set_ylabel('Percent Total Pulls',size=18)
        ax.set_xlabel('time',size=18)
        ax.set_title('Percent Pulls over time',size=25)
        ax.set_ylim(0,100)
        plt.legend(loc=0)

        if savefile != '':
            plt.savefig(savefile,dpi=150)

        plt.show()



def main():
    """
    Perform a test pipeline run using the default parameters.

    TODO: Add recording of total reward captured over time. Perhaps percent of reward captured over time.
    """

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--simulation", help="True or False to run a test simulation",
                           type=bool, default=False, required=False)
    argparser.add_argument("--levers_file", help="JSON file of synthetic lever information",
                           type=str, default=None, required=False)
    argparser.add_argument("--labelers_file", help="JSON file of synthetic labeler accuracy",
                           type=str, default=None, required=False)
    argparser.add_argument("--num_iter", help="How many iterations to run of bandit algorithm",
                           type=int, default=1000, required=False)
    argparser.add_argument("--save_png", help="Where to save the resulting .png",
                           type=str, default="test.png", required=False)
    argparser.add_argument("--run_id", help="The identifier string for this run of bandits",
                           type=str, default="test", required=True)
    argparser.add_argument("--continuation", help="Start from an ongoing run of bandits. True/False",
                           type=bool, default=False, required=False)
    argparser.add_argument("--one_iter", help="Most recent lever and score in 'lever_score' form",
                           type=str, default=None, required=False)
    argparser.add_argument("--parse_turk_file", help="Performs one iteration of model from data given by MTurk output filename",
                           type=str, default=None, required=False)


    args = argparser.parse_args()

    if args.simulation:
        """
        Run a simulation that uses synthetic data and synthetic labelers. You can
        specify the accuracy of each 'lever' in a .json file as a dictionary. Similarly,
        you can specify the accuracy of each 'labeler' in a .json file as a list.
        """
        if args.levers_file:
            with open(args.levers_file, 'r', encoding='utf-8') as f:
                levers = json.load(f)
        else:
            levers = {'a':0.27,'b':0.28,'c':0.29,'d':0.30}

        if args.labelers_file:
            with open(args.labelers_file, 'r', encoding='utf-8') as f:
                labelers = json.load(f)
        else:
            labelers = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

        x = SimulationPipeline(levers=levers, labelers=labelers)
        x.run_simulation(iterations=args.num_iter)
        x.plot_bandits(savefile=args.save_png)

    else:
        """
        Run bandits code on actual data.
        """

        x = Bandits(run_id=args.run_id, continuation=args.continuation, confidence_modifier=0.2)

        if args.one_iter:
            data = args.one_iter.split('_')
            nl = x.run_algorithm_once(data[0],float(data[1]))
            print(f"\n\nNext lever: {nl}")
            response = input(f'\n\nWould you like to record this in the ongoing log for {args.run_id}? (y/n)')
            if response == 'y':
                x.write_ucb()
                print('Updated log file.')
            else:
                print('Did not update log.')

            response = input(f"\n\nWould you like to see a summary? (y/n)")
            if response == 'y':
                print(x)
            else:
                print('No summary printed.')

        if args.parse_turk_file:

            lev = input('\n\nWhat is the name of this lever?\n')

            df = pd.read_csv(args.parse_turk_file)
            score, number_confident, number_labeled = determine_reward(df)

            print(f'\nReward calculated: {score}')

            nl = x.run_algorithm_once(lev,score)
            print(f"\n\nNext lever: {nl}")
            response = input(f'\n\nWould you like to record this in the ongoing log for {args.run_id}? (y/n)')
            if response == 'y':
                x.write_ucb()
                print('Updated log file.')
            else:
                print('Did not update log.')

            response = input(f"\n\nWould you like to see a summary? (y/n)")
            if response == 'y':
                print(x)
            else:
                print('No summary printed.')



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == '__main__':
    main()
