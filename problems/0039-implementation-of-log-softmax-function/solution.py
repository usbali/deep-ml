import numpy as np
import math
def log_softmax(scores: list) -> np.ndarray:
	
	#it calculate individual exp(x) and sum of all exp(x)
	#compute exponentials stabilized by subtracting max_score
	sum_exp =0
	max_score =max(scores)
	exp_scores = []
	for exp_score in scores:
		exp_value  = math.exp(exp_score - max_score)
		sum_exp +=exp_value
	#log softmax is  score - max_score - log of sum_exp
	log_soft = []
	for i,score in enumerate((scores)) :
		log_soft_value = (score - max_score) - math.log(sum_exp)
		log_soft.append(log_soft_value)	
	return np.array(log_soft)