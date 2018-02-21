import numpy as np
np.random.seed(1)

if __name__ == '__main__':

	with open('sampled_responses_branching1.txt', 'r') as handle:
		sampled1 = handle.readlines()
	with open('sampled_responses_branching2.txt', 'r') as handle:
		sampled2 = handle.readlines()
	with open('test_branching_contexts.txt', 'r') as handle:
		contexts = handle.readlines()

	indices = np.arange(len(contexts))	
	np.random.shuffle(indices)	

	wins = [0, 0, 0]
	for ix in indices:
		responses = [sampled1[ix], sampled2[ix]]

		r_indices = [0, 1]
		np.random.shuffle(r_indices)

		print 'Context:'
		print contexts[ix]

		print '0:'
		print responses[r_indices[0]].split('\t')[0]
		print '1:'
		print responses[r_indices[1]].split('\t')[0]

		best = raw_input('Which is better (2 tie):')
		if best.strip().lower() == 'quit':
			break
		elif best in ['0', '1', '2']:
			best = int(best)
			if best in [0, 1]: best = r_indices[best]
			wins[best] += 1

	print wins

