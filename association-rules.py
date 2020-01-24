import numpy as np
import pandas as pd
import os
import sys
import scipy
import random
from itertools import combinations
from itertools import permutations
import itertools


class apriori():
    def __init__(self, dataset, support_thresh, conf_thresh):
        """
            * INPUT PARAMETERS:
                * dataset = straightforward to understand what this is
                * support_thresh = our minimum threshold for support
                * conf_thresh = our confidence threshold for support
        """
        # set the input parameters as object stuff that we will keep track of
        self.s_threshold = support_thresh
        self.c_threshold = conf_thresh
        self.data = dataset

        # print(self.s_threshold)
        # print(self.c_threshold)
        # print(self.data)

        # init things that will hold our return values
        """
            * BRIEF DESCRIPTION OF OUR FREQ ITEMSET STORAGE: 
                *  f_itemsets[k] = contains a hashmap with all freq itemsets of size k+1
                *  this repeats across for all k 

            * BRIEF DESCRIPTION OF OUR ASSOC ITEMSET STORAGE 
                * similar setup to itemset storage
        """
        self.f_itemsets = []  # freq item sets
        self.assoc_rules = []  # assoc rules

    def generate_one_item(self):
        """
            * THE WHOLE POINT OF THIS MASSIVE TON OF BULLSHIT IS TO QUICKLY GET THE BEGINNING TABLE TO SPARK
            * OFF OUR BULLSHIT
        :return:
        """
        all_combos = {}
        perm_size = 1

        for index, row in self.data.iterrows():
            combos = list(combinations(row, perm_size))  # combination list for this dataframe

            for combo in combos:
                if combo in all_combos:
                    all_combos[combo] = all_combos[combo] + 1
                else:
                    if perm_size == 1:
                        # ah shit the table ain't be formed yet, go wild !!!!!
                        all_combos[combo] = 1

        all_combos_pruned = {}
        for k, v in all_combos.items():
            denom = len(self.data.index)
            act_support_val = v / denom

            # only if it meets the threshold will it be considered for the next iteration
            if act_support_val > self.s_threshold:
                # print(str(k) + ": " + str(v))
                all_combos_pruned[k] = v

        self.f_itemsets.append(all_combos_pruned)

        # generation if we can meet the support constraint add to our map
        one_two = {}
        for col in self.data.columns:
            act_support_val = (float(self.data[col].value_counts()[1])/len(self.data))

            if act_support_val >= self.s_threshold:
                one_two[col] = act_support_val

        # now generate the initial map we will use as well cause that's important to keep track of the boolean vals
        bool_map = {}
        for rows in range(len(self.data)):
            # now iterate through each row and get the true / false values
            bool_map[rows] = []

            # for every column in every row
            for col in self.data.columns:
                # if that [row][col] = TRUE
                if self.data.at[rows, col]:
                    bool_map[rows].append(col)

        return bool_map, one_two

    def generate_freq_data(self, bool_map, first_set):
        """
            * this is as simple as just generating a new table every time based on the last table
            * count the support for individual members of the table
            * drop those that don't garner enough support
            * that table will be final and continue through
        :return:
        """
        # GENERATE A NEW TABLE BECAUSE THE OLD ONE WON'T DO
        perm = 2

        # see here I create a list of hashmaps to store all lists for each permutation
        complete_freq_sets = []
        complete_freq_sets.append(first_set)

        old_table = self.f_itemsets[perm-2]

        while perm <= len(self.data.columns):
            # just continuing off of what I said above
            this_k_combo = list(combinations(first_set, perm))
            kth_freq_sets = {}

            # now iterate through the frequent itemset
            for combo in this_k_combo:
                ct = 0

                # now iterate through every row in the boolean map to check the original value of said row
                for row in bool_map:
                    # if this is in fact a match then it is a matching itemset a.k.a it is a subset
                    if set(combo).issubset(set(bool_map[row])):
                        ct = ct + 1

                act_support_val = ct / len(self.data)
                if act_support_val >= self.s_threshold:
                    kth_freq_sets[combo] = act_support_val

            if len(kth_freq_sets) != 0:
                complete_freq_sets.append(kth_freq_sets)
            else:
                break

            perm = perm + 1

        return complete_freq_sets

    def generate_association_rules(self, frequent_item_sets):
        # association set rules:
        # index reflects the true number etc assoc rule 1, assoc rule 2, etc.
        # where key(k) = anticedent, value(v) = consequent
        assoc_rules = []
        count = 0

        # for every k itemset
        for val in frequent_item_sets:
            # go through each individual itemset making up that k
            # print(val)
            for val_p in val:
                """
                    * Quicknotes: 
                        * so for each frequent itemset, get all of its subsets 
                        * subtract each subset from the total set and only select sets such that this difference is 1 
                            * e.g. (I - S) = 1 
                            * if the difference of (I - S) = 1 then that means that the consequent is 1 which is 
                                excellent 
                        * now get the support of I, and get the support of the S 
                        * divide the generated support of I, and the generated support of S 
                        * if division above > confidence threshold 
                            * include that association rule as a pair and store it
                """
                size_of_i = len(val_p)

                if size_of_i > 1 and not isinstance(val_p, str):
                    conv_t = list(itertools.chain(val_p))
                    # print(conv_t)

                    # for all of the size of i gen such combinations
                    combo_o_size = combinations(conv_t, (size_of_i - 1))

                    # get every one of the combinations you just generated
                    for combo in combo_o_size:
                        # now get the unique element because we'll have to get the support of this element
                        # print(combo)
                        unique_ele = [item for item in conv_t if item not in combo]
                        # print(unique_ele)

                        # got the unique element now get the support of each of these elements
                        # print(frequent_item_sets[0])
                        # print(type(combo))

                        to_check = combo
                        if len(combo) == 1:
                            to_check = to_check[0]

                        support_i = frequent_item_sets[len(combo)-1].get(to_check)
                        # print(support_s)
                        support_s = frequent_item_sets[size_of_i-1].get(val_p)
                        # print(support_i)

                        # compute the division above
                        division_above = float(support_s / support_i)

                        # if its greater than our confidence threshold increment said value
                        if division_above > self.c_threshold:
                            # count = count + 1
                            try:
                                assoc_rules[len(combo)] = assoc_rules[len(combo)] + 1
                            except IndexError:
                                assoc_rules.append(len(combo))

        conv_out = assoc_rules

        print()
        # for every association rule print
        for i in range(1, len(frequent_item_sets)):
            print("ASSOCIATION-RULES " + str(i + 1) + " " + str(assoc_rules[i]))

def count_num_itemsets(total_itemset):
    # so here we are going to count the number of itemsets in the final version:
    counts = {}
    for k in total_itemset:
        for key, value in k.items():
            # i was getting an error with numbers so I'm going to have to catch that error
            if isinstance(key, str):
                try:
                    counts[1] = counts[1] + 1
                except KeyError:
                    counts[1] = 1
            else:
                try:
                    counts[len(key)] = counts[len(key)] + 1
                except KeyError:
                    counts[len(key)] = 1

    # now just go ahead and print the generated itemset
    indiv_count_access = counts.keys()
    for indiv_count in sorted(indiv_count_access):
        print("FREQUENT-ITEMS " + str(indiv_count) + " " + str(counts[indiv_count]))

# TODO FIX PREPROCESSING
def PreProcess(dataset):
    """
        *  DESCRIPTION:
            * this dataset simply subsets the entire dataset, so ya, lets eshkitit

        *  INPUT PARAMETERS:
            * dataset: this is just the unaltered dataset

        *  OUTPUT PARAMETERS:
            * actual_df: this is a dataframe representing subsetted dataset. expanded following the rules described in
              the description
    """
    temp_df = pd.read_csv(dataset)

    # goodForGroups replacement
    temp_df['goodForGroups'] = temp_df['goodForGroups'].replace(0, 'goodForGroups_0')
    temp_df['goodForGroups'] = temp_df['goodForGroups'].replace(1, 'goodForGroups_1')

    # open replacement
    temp_df['open'] = temp_df['open'].replace(True, 'open_TRUE')
    temp_df['open'] = temp_df['open'].replace(False, 'open_FALSE')

    # delivery replacement
    temp_df['delivery'] = temp_df['delivery'].replace(True, 'delivery_TRUE')
    temp_df['delivery'] = temp_df['delivery'].replace(False, 'delivery_FALSE')

    # waiterService replacement
    temp_df['waiterService'] = temp_df['waiterService'].replace(True, 'waiterService_TRUE')
    temp_df['waiterService'] = temp_df['waiterService'].replace(False, 'waiterService_FALSE')

    # caters replacement
    temp_df['caters'] = temp_df['caters'].replace(True, 'caters_TRUE')
    temp_df['caters'] = temp_df['caters'].replace(False, 'caters_FALSE')



    # do a different dataframe thats better
    temp_df = pd.get_dummies(temp_df)

    new_df = pd.read_csv(dataset)
    new_df = new_df.astype(str)

    for col in new_df.columns:
        if col == 'goodForGroups':
            val = new_df[col].str.contains(pat='1')
            new_df[col] = val
            continue

        if col == 'open':
            val = new_df[col].str.contains(pat='True')
            new_df[col] = val
            continue

        if col == 'delivery':
            val = new_df[col].str.contains(pat='True')
            new_df[col] = val
            continue

        if col == 'waiterService':
            val = new_df[col].str.contains(pat='True')
            new_df[col] = val
            continue

        if col == 'caters':
            val = new_df[col].str.contains(pat='True')
            new_df[col] = val
            continue

        for val in new_df[col].unique():
            new_df[col + '-' + str(val)] = new_df[col]
            new_df[col + '-' + str(val)] = new_df[col].str.contains(pat=val)

        del [col]

    del new_df['attire']
    del new_df['priceRange']
    del new_df['city']
    del new_df['noiseLevel']
    del new_df['state']
    del new_df['alcohol']
    return new_df


if __name__ == '__main__':
    """
        * DESCRIPTION:
            * aight this is the last hw so the comments are going to be pretty light, we take in the desired parameters, 
                get the frequent itemsets, from the frequent itemsets generate association rules and then return and 
                print that shit

        * INPUT PARAMETERS: 
            *argv[1] = yelp5.csv / any desired training set matching the input data format  
            *argv[2] = minsup; this is the value of the minimum support threshold we are going to use (FREQ. ITEMSETS)  
            *argv[3] = minconf; this is the value of the minimum confidence threshold we are going to use (ASSOC. RULES) 
    """

    # define the input arguments
    data = sys.argv[1]
    support_thresh = sys.argv[2]
    conf_thresh = sys.argv[3]

    # first step: let's preprocess the data
    data_set = PreProcess(data)
    # print(data_set)

    # second step: let's init apriori
    a_p = apriori(data_set, float(support_thresh), float(conf_thresh))

    # third step let's find the first set of rules and some setup
    results_one = a_p.generate_one_item()
    bool_map = results_one[0]
    first_set = results_one[1]

    # fourth step calculate all frequent item sets
    all_frequents = a_p.generate_freq_data(bool_map, first_set)

    # fifth step print the frequent item sets we got
    count_num_itemsets(all_frequents)

    # sixth step generate association rules based on the calculated frequent item sets
    a_p.generate_association_rules(all_frequents)

    # a_p.generate_freq_data()
    # print(all_frequents)

    # fourth step print!!!!!!!!!!!!!!!!!!!!!!!!!!
