# -*- coding: utf-8 -*_
from DecisionTree import DecisionTree

dt = DecisionTree(training_datafile ="./jueceshu.data")
dt.get_training_data()
dt.show_training_data()
root_node = dt.construct_decision_tree_classifier()
root_node.display_decision_tree("   ")
test_sample = ['exercising=>never', 'smoking=>heavy', 'fatIntake=>heavy', 'videoAddiction=>heavy']
classsification = dt.classify(root_node,test_sample)
print classsification
