###### Document to create and test dummy MTurk results data #######

import pandas as pd
import numpy as np
from scipy.stats import uniform, norm, gamma, binom
import datetime

# column names from MTurk results CSV:

# HITId	HITTypeId	Title	Description
#Keywords	Reward	CreationTime	MaxAssignments	RequesterAnnotation
#AssignmentDurationInSeconds	AutoApprovalDelayInSeconds
#Expiration	NumberOfSimilarHITs	LifetimeInSeconds	AssignmentId	WorkerId
#AssignmentStatus	AcceptTime	SubmitTime	AutoApprovalTime	ApprovalTime
#RejectionTime	RequesterFeedback	WorkTimeInSeconds	LifetimeApprovalRate
#Last30DaysApprovalRate	Last7DaysApprovalRate	Input.TEXT
#Answer.Directive Sentence	Answer.no.0	Answer.yes.1	Approve	Reject

# columns of importance:
#Input.TEXT: This contains the input sentence/HIT sentence
#Answer.Directive Sentence: This contains text from the input sentence, for what section of text the
#MTurker consider a directive. Could be NULL (no sentence provided)
#Answer.no.0: Either TRUE or FALSE boolean, if the MTurker said no directive was found in HIT
#Answer.yes.1: Either TRUE or FALSE boolean, if the MTurker said one or more directives was found in HIT


##### Checking that MTurk data returns as booleans (it does) #######

#mturk_data = pd.read_csv('Batch_3949723_batch_results.csv')

#mturk_dataå[['Answer.no.0', 'Answer.yes.1']].iloc[0][0] == 1


###### Develop Ground Truth Data ###########

ground_truth_labels = binom.rvs(n=1,p=0.9,size=100)


##### Export ground_truth labels as separate txt file

np.savetxt('test_gt.txt', ground_truth_labels, delimiter=',', fmt = '%5.0f')


######## create a function that is the accuracy rate of MTurker

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



##### Testing that MTurker accuracy rate and label prediction rate are close to equal (they are). Test is on multiple MTurk function #######


test_Mturker = multiple_MTurk_accuracy(0.9, ground_truth_labels, 5)

len(test_Mturker)

len(np.where(ground_truth_labels == test_Mturker[3])[0])


####### Create dummy data CSV file that mimics MTurk data export file #######


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

    time_stamp = f"{datetime.datetime.now().year}-{datetime.datetime.now().month}-{datetime.datetime.now().day}-{datetime.datetime.now().hour}{datetime.datetime.now().minute}"

    fake_MTurk_result_file.to_csv(f"fake_MTurk_results_{time_stamp}.csv", index = False)

    return fake_MTurk_result_file


### Test on mutliple MTurkers at same accuracy rate ###

fake_MTurk_results_file_generator(test_Mturker).head(n = 5)


### Test on mutliple MTurkers at different accuracy rates ###

fake_MTurk_results_file_generator(
    different_rate_Mturkers([0.1, 0.5, 0.8, 0.3, 0.01, 0.012, 0.6, 0.75], ground_truth_labels)).head(n = 5)
