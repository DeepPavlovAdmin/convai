"""Beam search implementation in PyTorch."""
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

# Code borrowed from PyTorch OpenNMT example
# https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Beam.py

# Hwaran Lee, KAIST: 2017-present

import torch
import pdb

debug=False

class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, vocab, cuda=False):
        """Initialize params."""
        ## vocab = self.dict.tok2ind
        self.size = size
        self.done = False
        self.pad = vocab['__NULL__']
        self.bos = vocab['__START__']
        self.eos = vocab['__END__']
        self.unk = vocab['__UNK__']
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.score_mask = self.tt.FloatTensor(size).fill_(1)

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

        # The attentions (matrix) for each time.
        self.attn = []
        
        # The 'done' for each translation on the beam.
        self.doneYs = [False]*size
        
        self.active_idx_list = list(range(self.size))
        self.active_idx = self.tt.LongTensor(self.active_idx_list)
        
        # Generating unk token or not
        self.gen_unk = False

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance(self, word_lk):
        """Advance the beam."""
        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0]

        flat_beam_lk = beam_lk.view(-1)
        
        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
            
        return self.done

    def advance_end(self, word_lk):
        """Advance the beam.
            Until each beam meets __eos__
            Do not generate __unk__
        """
        num_words = word_lk.size(1)
        
        if debug:
            print("score mask")
            print(self.score_mask)
        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = self.score_mask.unsqueeze(1).expand_as(word_lk)*word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
            beam_lk = beam_lk.index_select(0, self.active_idx)        
        else:
            beam_lk = word_lk[0]

        # Avoid generating UNK token
        if not self.gen_unk:
            if beam_lk.dim() == 1:
                beam_lk[self.unk] = -100
            else:
                beam_lk[:, self.unk] = -100                  
        
        ## self.score_mask --> exclude the row
        ## and sorting        
        flat_beam_lk = beam_lk.view(-1)
        bestScores, bestScoresId = flat_beam_lk.topk(len(self.active_idx_list), 0, True, True) ## self.size
        self.scores.scatter_(0, self.active_idx, bestScores)
        
        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        next_ys = bestScoresId - prev_k * num_words
        
        if self.tt == torch.cuda:
            prev_k1 = torch.arange(0,self.size).long().cuda().scatter_(0, self.active_idx, self.active_idx[prev_k])        
        else:
            prev_k1 = torch.arange(0,self.size).long().scatter_(0, self.active_idx, self.active_idx[prev_k])      
                    
        next_ys1 = self.tt.LongTensor(self.size).fill_(self.pad).scatter_(0, self.active_idx, next_ys)

        self.prevKs.append(prev_k1) # trasform prev_k => original index
        self.nextYs.append(next_ys1)

        # mask
        done = True
        for i in range(self.size):
            if self.nextYs[-1][i]  == self.eos:
                self.doneYs[i] = True
                self.score_mask[i] = 0
                self.active_idx_list.remove(i)
                if debug:
                    pdb.set_trace()
                    print(i)
                    print(self.active_idx_list)
            done *= self.doneYs[i]
        
        self.active_idx = self.tt.LongTensor(self.active_idx_list)        
        self.done = done
        return self.done    
    
    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        #print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]
