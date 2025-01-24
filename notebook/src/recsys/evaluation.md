# Evaluation

Evaluation is a non-trivial topic for recsys, and different approaches measure different things. Suppose we have a dataset of user-item interactions with a timestamp. 

<<Random splitting>> simply takes a random split of say 75% for train and 25% for test. The problems with this approach:
- No guarantee of overlap in users across train and test. If a user does not appear in the train set, it is not possible to recommend items for him/her in the test set.
- Chronological overlap between train and test set, leading to data leakage issues.

<<Stratified splitting>> addresses the user overlap issue by ensuring that the number of rows per user in the train and test set are approximately 75% and 25% of the number of rows in the original data respectively. This ensures that we have sufficient training and test data for each user, so that the collaborative filtering algorithm has a fair chance of recommending items for each user.

However, stratified splitting still involves randomly assigning rows for each user into the train and test set, which does not address the chronological overlap issue. <<Temporal stratified splitting>> addresses this issue by assigning the 75% and 25% of train and test data based on chronological order. In other words, the oldest 75% of data for each user is assigned to the train set.

The extreme version of temporal stratified splitting is <<leave last out>> splitting, in which all but the latest row for each user is put into the train set. This is suitable for settings where the task is to predict the very next action which the user will take (e.g. which song will the user listen to next).

Note that temporal stratified splitting may potentially introduce temporal overlap between the train and test sets across users. That is, the train set period for user A may potentially overlap with the test set period for user B. Hence, if there are strong concerns with temporal effects in the dataset, we may need to be mindful of this.

<<Global temporal splitting>> addresses this issue by assigning the oldest 75% of data across all users to the train set. This addresses the data leakage issue and more closely resembles actual production setting. However, there is no guarantee on the amount of train/test data for each user. Hence we may need to drop rows where there exists test data for user A but no corresponding train data due to the global temporal cutoff.


