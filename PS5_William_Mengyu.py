import csv
import math
import numpy as np
import collections
import pprint

"""************************************************************************
* function:  load_data(for_prediction)
* arguments:
*       -for_prediction: boolean that is "True" if reading in unlabeled data for
*                        making predictions (i.e., 'loans_B_unlabeled.csv'),
*                        "False" if reading in labeled data to build a 
*                        regression tree (i.e., 'loans_A_labeled.csv')
* return value:  a list of tuples representing the labeled loans data if
*                `for_prediction` is False; a list of dictionaries to represent
*                features of unlabeled loans data if `for_prediction` is True
*
* TODO:  Read in the loans data from the provided csv file. If `for_prediction`
*        is False, store the observations as a list of tuples (a, b), where 'a'
*        is a dictionary of features and 'b' is the value of the
*        `days_until_funded` variable. If `for_prediction` is True, store the
*        observations as a list of dictionaries where each dictionary is that
*        of features.
************************************************************************"""


def load_data(for_prediction=False):
    file = "loans_AB_labeled.csv"
    if for_prediction:
        file = "loans_ABC_unlabeled.csv"
    data = []

    with open(file, "rt", encoding="utf8") as f:
        dict_reader = csv.DictReader(f)
        for observation in dict_reader:
            feature_dict = {}

            for key, value in observation.items():
                feature_dict[key] = value

            if for_prediction == False:
                data.append((feature_dict, float(observation["days_until_funded"])))

            elif for_prediction == True:
                data.append(feature_dict)
    return data


def create_bins(data):
    new_data = []
    for item in data:
        # Check if item is a tuple (labeled data) or a dictionary (unlabeled data)
        if isinstance(item, tuple):
            features, value = item
        else:
            features = item
            value = None  # or some default value if needed

        # create loan_amount bins
        amount = float(features["loan_amount"])
        if amount <= 500:
            loan_bin = "Loan Amount (<=500)"
        elif amount <= 1000:
            loan_bin = "Loan Amount (501-1000)"
        elif amount <= 2000:
            loan_bin = "Loan Amount (1001-2000)"
        else:
            loan_bin = "Loan Amount (>=2000)"

        features["loan_amount_bin"] = loan_bin

        # create repayment_term bins
        repay_term = float(features["repayment_term"])
        if repay_term <= 5:
            term_bin = "Repayment Term (<=5)"
        elif repay_term <= 10:
            term_bin = "Repayment Term (6-10)"
        elif repay_term <= 15:
            term_bin = "Repayment Term (11-15)"
        else:
            term_bin = "Repayment Term (15-20)"

        features["repayment_term_bin"] = term_bin

        # create posted_date bins
        posted_date = features["posted_date"]
        year = int(posted_date[:4])
        if year < 2008:
            date_bin = "Posted Date (<2008)"
        elif 2008 <= year <= 2009:
            date_bin = "Posted Date (2008-2009)"
        elif 2010 <= year <= 2011:
            date_bin = "Posted Date (2010-2011)"
        elif 2012 <= year <= 2013:
            date_bin = "Posted Date (2012-2013)"
        else:
            date_bin = "Posted Date (>2013)"
        features["posted_date_bin"] = date_bin

        if value is not None:
            new_data.append((features, value))
        else:
            new_data.append(features)
    return new_data


def group_feature_avgs(data, features):
    # data: a list of tuples of dict where 'a' is features and 'b' is days_until_funded
    # returns a dict mapping each feature chosen to its groupby averages
    # Example return structure:{"sector": {"Retail": 10.5, "Agriculture": 7.2, ... }, "activity": {"Clothing": 9.0, "Farming": 8.3, ...}, ...

    group_averages = {}
    for feature in features:
        groups = {}

        for feature_dict, target in data:  # feature_dict = 'a', target = 'b'
            target_value = float(target)
            key = feature_dict.get(feature)  # get the value for current feature

            if key in groups:
                groups[key].append(target_value)

            else:
                groups[key] = [target_value]

        # compute the average for each group for this feature.
        averages = {k: sum(v) / len(v) for k, v in groups.items()}
        group_averages[feature] = averages

    return group_averages


"""************************************************************************
* function:  continuous_to_percentile(observations, continuous_var, n_bins)
* arguments:
*       -observations: a list of tuples (a, b) representing loans data, where
*                      'a' is a dictionary of features and 'b' is the value of
*                      the `days_until_funded` variable
*       -continuous_var: string representing the feature whose values are
*                        considered continuous rather than binary or
*                        categorical
*       -n_bins: integer to indicate how many "bins" of percentiles should be
*                created for the continuous variable. For instance, if it is
*                4, the function will create quartiles (i.e., 0-25th, 25-50th,
*                50-75th, and 75-100th percentiles) and create a binary
*                variable each for a quartile that equals to 1 if it falls
*                within the said quartile.
* 
* example use case:
*     # assuming `load_data` function was written correctly
*     data = load_data()
*     
*     # example with the variable "loan_amount"
*     modified_data = continuous_to_percentile(data, "loan_amount", 4)
*
* return value:  a list of tuples representing the loans data, but with the
*                specified 'continuous_var' replaced by binary variables
*                as written in the description for 'n_bins'.
************************************************************************"""


def continuous_to_percentile(observations, continuous_var, n_bins=4):
    var_values = [float(obs[0][continuous_var]) for obs in observations]
    percentiles_to_calc = [(i + 1) * 100 / n_bins for i in range(n_bins)]
    percentiles = np.percentile(var_values, percentiles_to_calc)
    new_var_names = [f"{continuous_var}_{i + 1}_{n_bins}" for i in range(n_bins)]

    new_data = []
    for obs in observations:
        features, days_until_funded = obs
        features_to_modify = features.copy()
        var_value = float(features_to_modify.pop(continuous_var))
        for var in new_var_names:
            features_to_modify[var] = 0
        for i in range(n_bins):
            var_name = new_var_names[i]
            lower = percentiles[i - 1] if i > 0 else -float('inf')
            upper = percentiles[i]
            if lower < var_value <= upper:
                features_to_modify[var_name] = 1
                break
        new_data.append((features_to_modify, days_until_funded))
    return new_data


"""************************************************************************
* function: partition_loss(subsets)
* arguments:
* 		-subsets:  a list of lists of labeled data (representing groups
				   of observations formed by a split)
* return value:  loss value of a partition into the given subsets
*
* TODO: Write a function that computes the loss of a partition for
*       given subsets
************************************************************************"""


def partition_loss(subsets):
    total_loss = 0
    for subset in subsets:
        if not subset:
            continue
        targets = []
        for obs in subset:
            targets.append(obs[1])
        mean = sum(targets) / len(targets)
        subset_loss = sum((t - mean) ** 2 for t in targets)
        total_loss += subset_loss
    return total_loss


"""************************************************************************
* function: partition_by(inputs, attribute)
* arguments:
* 		-inputs:  a list of observations in the form of tuples
*		-attribute:  an attribute on which to split
* return value:  a list of lists, where each list represents a subset of
*				 the inputs that share a common value of the given 
*				 attribute
************************************************************************"""


def partition_by(inputs, attribute):
    groups = collections.defaultdict(list)
    for input_ in inputs:
        key = input_[0][attribute]
        groups[key].append(input_)
    return groups


"""************************************************************************
* function: partition_loss_by(inputs, attribute)
* arguments:
* 		-inputs:  a list of observations in the form of tuples
*		-attribute:  an attribute on which to split
* return value:  the loss value of splitting the inputs based on the
*				 given attribute
************************************************************************"""


def partition_loss_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    return partition_loss(partitions.values())


"""************************************************************************
* function:  build_tree(inputs, num_levels, split_candidates=None)
*
* arguments:
* 		-inputs:  labeled data used to construct the tree; should be in the
*				  form of a list of tuples (a, b) where 'a' is a dictionary
*				  of features and 'b' is a label
*		-num_levels:  the goal number of levels for our output tree
*		-split_candidates:  variables that we could possibly split on.  For
*							our first level, all variables are candidates
*							(see first two lines in the function).
*			
* return value:  a tree in the form of a tuple (a, b) where 'a' is the
*				 variable to split on and 'b' is a dictionary representing
*				 the outcome class/outcome for each value of 'a'.
* 
* TODO:  Write a recursive function that builds a REGRESSION tree (NOT a
*        classification tree!) of the specified number of levels based on
*        labeled data "inputs"
************************************************************************"""


def build_tree(inputs, num_levels, split_candidates=None):
    if split_candidates is None:
        split_candidates = list(inputs[0][0].keys()) if inputs else []
    split_candidates = list(split_candidates)

    # base case: if reached desired depth of tree or more attributes to split
    if num_levels <= 0 or not split_candidates:
        targets = [obs[1] for obs in inputs]  # targets is a list of labels
        return sum(targets) / len(targets) if inputs else 0

    targets = [obs[1] for obs in inputs]
    if len(set(targets)) == 1:  # if all labels are the same, return the same value
        return targets[0]

    best_attr = None
    min_loss = float('inf')
    for attr in split_candidates:  # choose best attr to split on
        loss = partition_loss_by(inputs, attr)
        if loss < min_loss:
            min_loss = loss
            best_attr = attr

    #    if best_attr is None: # if no best att, return avg
    #        return sum(targets) / len(targets) if inputs else 0

    partitions = partition_by(inputs, best_attr)
    new_split_candidates = [sc for sc in split_candidates if sc != best_attr]

    subtree_dict = {}
    for value, subset in partitions.items():
        subtree = build_tree(subset, num_levels - 1, new_split_candidates)
        subtree_dict[value] = subtree

    return (best_attr, subtree_dict)


"""************************************************************************
* function:  predict(tree, to_predict)
*
* arguments:
* 		-tree:  a tree built with the build_tree function
*		-to_predict:  a dictionary of features
*
* return value:  a value indicating a prediction of days_until_funded

* TODO:  Write a recursive function that uses "tree" and the values in the
*		 dictionary "to_predict" to output a predicted value.
************************************************************************"""


def predict(tree, to_predict):
    if not isinstance(tree, tuple):
        return tree  # end of recursion
    attr, subtree_dict = tree
    # attr is the feature which node splits
    # subtree_dict is a dict which keys are different values of attr
    value = to_predict.get(attr)
    if value in subtree_dict:
        return predict(subtree_dict[value], to_predict)  # if value in subtree, predict that branch
    else:
        predictions = []
        for subtree in subtree_dict.values():  # if value is new, gather predictions from every branch in subtree
            pred = predict(subtree, to_predict)
            predictions.append(pred)
        return sum(predictions) / len(predictions) if predictions else 0


def mse(labels):  # calculate mse
    labels = [float(label) for label in labels]
    if not labels:
        return 0
    mean_label = sum(labels) / len(labels)
    return sum((label - mean_label) ** 2 for label in labels) / len(labels)


def bootstrap_sample(inputs, length):
    """
    Takes a list of inputs from the training data and a length,
    and returns a bootstrap sample of that length from the list of inputs.

    Parameters
    ----------
    inputs : a collection of tuples (a, b) where a is the feature dictionary
    and b is the lavel (days_until_funded) for each loan
    length : a int for the desired length of the bootstrap sample

    Returns
    ----------
    (a list of tuples)
    a bootstrap sample (in the form of a list) of param length from the list of inputs
    """
    a = len(inputs)
    indices = range(a)
    sampled_indices = np.random.choice(indices, size=length)
    return [inputs[i] for i in sampled_indices]


def build_forest_tree(inputs, num_levels, num_split_candidates, split_candidates=None):
    """
    TODO: takes a list of inputs (in the form of tuples), the
    number of levels (num_levels) to use in building a tree, and the number of
    split candidates (num_split_candidates) to randomly choose at each split in
    the tree

    Parameters
    ----------
    inputs : TODO: list
        TODO: A list of tuples, where each tuple contains a dictionary
        of feature values and a label.
    num_levels : TODO: int
        TODO:  The depth of the tree,
        controls how many splits are made before stopping.
    num_split_candidates : TODO: int
        TODO:  The number of random attributes to
        consider for splitting at each node, improving diversity.
    split_candidates : TODO: list, optional
        TODO: The list of available attributes for splitting,
        defaults to all attributes in the dataset.

    Returns
    -------
    TODO: SHORT DESCRIPTION AND TYPE.

    """
    # if first pass, all keys are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()
        split_candidates = list(split_candidates)

    if len(split_candidates) <= num_split_candidates:
        sampled_split_candidates = split_candidates
    else:
        sampled_split_candidates = random.sample(split_candidates, num_split_candidates)

    if num_levels == 0:
        return sum([x[1] for x in inputs]) / float(len(inputs))

    # if no split candidates left, return the average
    if not split_candidates:
        return sum([x[1] for x in inputs]) / float(len(inputs))

    # otherwise, split on best attribute
    best_attribute = random.choice(sampled_split_candidates)
    best_loss = partition_loss_by(inputs, best_attribute)
    for candidate in sampled_split_candidates:
        e = partition_loss_by(inputs, candidate)
        if e < best_loss:
            best_attribute = candidate
            best_loss = e

    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates if a != best_attribute]

    subtrees = {
        attribute_value: build_forest_tree(
            subset, num_levels - 1, num_split_candidates, new_candidates
        )
        for attribute_value, subset in partitions.items()
    }
    if len(partitions) != 1:
        subtrees = {
            attribute_value: build_forest_tree(
                subset, num_levels - 1, num_split_candidates, new_candidates
            )
            for attribute_value, subset in partitions.items()
        }
        return (best_attribute, subtrees)

    else:
        return sum([x[1] for x in inputs]) / float(len(inputs))


def classify(tree, to_classify):
    if isinstance(tree, float):
        return tree
    else:
        attribute, subtree_dict = tree
        value = to_classify.get(attribute)
        if value in subtree_dict:
            return classify(subtree_dict[value], to_classify)
        else:
            # Value not seen during training: average over all branches
            predictions = [classify(sub, to_classify) for sub in subtree_dict.values()]
            return sum(predictions) / len(predictions) if predictions else 0


def forest_classify(trees, loan):
    """
    Uses an ensemble of decision trees to classify an input instance by averaging predictions.
    Each tree independently predicts the outcome, and the final classification is determined by
    averaging their outputs to reduce variance and improve accuracy.

    Parameters
    ----------
    trees : TYPE-list
    A list of decision trees forming the random forest.
    loan : TYPE-dict
    A dictionary representing a loan application with its attributes.

    Returns
    -------
    float, the averaged predicted value from the decision trees, providing a more robust estimate.
    """
    votes = [classify(tree, loan) for tree in trees]
    return sum(votes) / float(len(votes))


"""********************************
main function:


********************************"""
import random

if __name__ == "__main__":

    """
    ############################### predicting on loans_A_labeled ###############################
    loans_A_labeled = load_data(for_prediction=False) 
    loans_A_labeled_bins = create_bins(loans_A_labeled)
    select_feature_avgs = group_feature_avgs(loans_A_labeled_bins, features=["sector", "activity", "loan_amount_bin", "repayment_term_bin"])

    #print("Group Feature Averages:")
    #print(select_feature_avgs)

    random.seed(42)
    random.shuffle(loans_A_labeled)

    decision_tree = build_tree(loans_A_labeled_bins, num_levels=8, split_candidates=["sector", "activity", "loan_amount_bin", "repayment_term_bin"])

    predictions = [predict(decision_tree, features) for features, target in loans_A_labeled_bins]
    actuals = [target for features, target in loans_A_labeled_bins]

    test_mse = mse([actual - pred for actual, pred in zip(actuals, predictions)])
    print("Test MSE, Loans_A_Labeled:", test_mse)


    ############################### predicting on loans_AB_labeled ###############################
    loans_AB_labeled = load_data(for_prediction=False) 
    loans_AB_labeled_bins = create_bins(loans_AB_labeled)
    select_feature_avgs = group_feature_avgs(loans_AB_labeled_bins, features=["sector", "activity", "loan_amount_bin", "repayment_term_bin"])

    #print("Group Feature Averages:")
    #print(select_feature_avgs)

    random.seed(42)
    random.shuffle(loans_A_labeled)

    decision_tree = build_tree(loans_AB_labeled_bins, num_levels=8, split_candidates=["sector", "activity", "loan_amount_bin", "repayment_term_bin"])

    predictions = [predict(decision_tree, features) for features, target in loans_AB_labeled_bins]
    actuals = [target for features, target in loans_AB_labeled_bins]

    test_mse = mse([actual - pred for actual, pred in zip(actuals, predictions)])
    print("Test MSE, Loans_AB_Labeled:", test_mse)




    ############################### Training on loans_A_labeled ###############################
    # loads data into a list of tuples
    loans_A_labeled = load_data(for_prediction=False) 

    # create additional bins for "loan_amount", "repayment term" for load_data
    loans_A_labeled_bins = create_bins(loans_A_labeled)

    # groupby avgs for select features
    select_feature_avgs = group_feature_avgs(loans_A_labeled_bins, features=["sector", "activity", "loan_amount_bin", "repayment_term_bin"])

    random.seed(42)
    random.shuffle(loans_A_labeled)

    # splits data by split_index
    split_index = int((len(loans_A_labeled_bins)) *0.67)
    training_data = loans_A_labeled_bins[:split_index]
    test_data = loans_A_labeled_bins[split_index:]

    # extracting features and labels
    test_features     = [item[0] for item in test_data]
    test_targets      = [item[1] for item in test_data]

    decision_tree = build_tree(training_data, num_levels=2, split_candidates=["sector"])

    # MSE on A2
    test_predictions = [predict(decision_tree, features) for features in test_features]
    test_mse = mse([(actual - predicted) for actual, predicted in zip(test_targets, test_predictions)])

    print("Test MSE, Loans_A_Labeled:", test_mse)
    """

    ############################### Random Forest ###############################
    loans_AB_labeled = load_data(for_prediction=False)
    loans_AB_labeled_bins = create_bins(loans_AB_labeled)

    num_trees = 200
    num_levels = 5
    num_split_candidates = 2

    forest = []

    for i in range(num_trees):
        bs = bootstrap_sample(loans_AB_labeled_bins, 1000)
        tree = build_forest_tree(bs, num_levels, num_split_candidates,
                                 split_candidates=["sector", "activity", "loan_amount_bin", "repayment_term_bin"])
        forest.append(tree)

    # predict days_until_funded for each loan in the labeled data
    predctions = []
    actuals = []
    for features, target in loans_AB_labeled_bins:
        pred = forest_classify(forest, features)
        predctions.append(pred)
        actuals.append(target)

    test_mse = mse([actual - pred for actual, pred in zip(actuals, predctions)])
    print("Random Forest MSE, Loans_AB_Labeled:", test_mse)


# ------------------- CREATE LOANS_ABC_PREDICTED.CSV FILE -----------------
    def save_predictions(filename, forest, loans):
        with open(filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "days_until_funded_MD_WW"])
            for loan in loans:
                prediction = forest_classify(forest, loan)
                writer.writerow([loan["id"], prediction])

    # generate the predicted file
    loans_ABC_unlabeled = load_data(for_prediction=True)
    save_predictions("loans_ABC_predicted.csv", forest, loans_ABC_unlabeled)

print("Predictions saved to loans_ABC_predicted.csv")
