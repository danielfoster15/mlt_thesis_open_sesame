from collections import Counter
model = 'argid_no_data_fes'
gold_standard = 'fn1.7.dev.syntaxnet_no_data_fes.conll'


with open('logs/'+model+'/predicted-1.7-argid-dev.conll', 'r') as f:
	predicted_FEs =[]
	for line in f:
		if line != '\n':
			predicted_FE = line.split()[-1]
			predicted_FEs.append(predicted_FE)

with open('data/neural/fn1.7/'+gold_standard, 'r') as f:
	gold_FEs =[]
	for line in f:
		if line != '\n':
			gold_FE = line.split()[-1]
			gold_FEs.append(gold_FE)

recall_problems = []
precision_problems = []
for gold, prediction in list(zip(gold_FEs, predicted_FEs)):
	if gold != prediction:
		if prediction == 'O':
			recall_problems.append(gold)
		else:
			precision_problems.append((gold, prediction))



print(Counter(recall_problems).most_common())
print(Counter(precision_problems).most_common())