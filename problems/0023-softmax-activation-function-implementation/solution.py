import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    # Need to return the list of softmax prob
#     sum_exp = 0
#    #Calculate the exponential first
#     exp_scores = []
#     for score in scores:
#     #calculate individual exponential values
#         exp_values = math.exp(score)
#         exp_scores.append(exp_values)
#         sum_exp +=exp_values

#     # calculate softmax value in the list
#     soft_op = []
#     for exp_score in exp_scores:
#         soft_op.append(exp_score/sum_exp)
#     return soft_op  

# Issue of overflow is solved
#    softmax = math.exp(z - max(scores))/()
    sum_exp = 0
    max_score = max(scores)
   #Calculate the exponential first
    exp_scores = []
    for score in scores:
    #calculate individual exponential values
        exp_values = math.exp(score - max_score)
        exp_scores.append(exp_values)
        sum_exp +=exp_values

    # calculate softmax value in the list
    soft_op = []
    for exp_score in exp_scores:
        soft_op.append(exp_score/sum_exp)
    return soft_op   