Plan:

1. create random dataset
2. train stateful LSTM:
input (pool based sampling, 20 random data points at a time, per data point each:
- proba of classifier
(- cluster density measures)
(- dominant label amoung density (?))
(- proposed label by classifier)


Output:
- take label by classifier
- alternatively ask for label for sample (but not for all)
(-take label by cluster region)

true_y aka IL is then: 
- if proposed label by classifier is correct (yes, no, it is better to take the correct one, really bad if a wrong one is taken, and semi ok if nothing gets taken)
- correct ids of those samples, who benefit the most, error is larger depending on the peaked accuracy the samples 

open questions:
- how to take budget cost etc. into error into account? do i need that?
- add fixed penalty for asked labeled samples, less penalty if the label of the classifier is used instead?
