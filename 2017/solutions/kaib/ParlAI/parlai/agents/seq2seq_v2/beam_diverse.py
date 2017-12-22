"""Diverse Beam search implementation in PyTorch."""
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

    def __init__(self, size, vocab, group=10, hold_step=1, cuda=False):
        """Initialize params."""
        ## vocab = self.dict.tok2ind
        self.size = size
        self.group = group
        assert size%group == 0, 'Beam: size % group == 0'
        self.size_g = int(size/group)
        self.hold_step = hold_step
        
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
        
        # self.size_g, self.group 
        self.active_idx_list = [] #len of self.group
        self.active_idx = [] # tensor list len of self.group, each tensor is size of self.size_g 
        for g in range(self.group):
            self.active_idx_list.append(list(range(self.size_g)))
            self.active_idx.append(self.tt.LongTensor(self.active_idx_list[g]))        

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

    def group_advance(self, word_lk, scores, score_mask, active_idx, active_idx_list):
        # beam inside each group
        # word_lk: #beam_g x #V
        # score: #beam_g x 1
        
        num_words = word_lk.size(1)
        
        if debug:
            print("score mask")
            print(score_mask)
        
        if len(active_idx_list)>0:    
            # Sum the previous scores.
            if len(self.prevKs) > 1:
                beam_lk = score_mask.unsqueeze(1).expand_as(word_lk)*word_lk + scores.unsqueeze(1).expand_as(word_lk) 
                beam_lk = beam_lk.index_select(0, active_idx)        
            else:
                #beam_lk = word_lk[0] ## NEEDS?
                beam_lk = word_lk[0] + scores[0] ## NEEDS?

            # Avoid generating UNK token
            if not self.gen_unk:
                if beam_lk.dim() == 1:
                    beam_lk[self.unk] = -100
                else:
                    beam_lk[:, self.unk] = -100               
    
            ## self.score_mask --> exclude the row and sorting
            flat_beam_lk = beam_lk.view(-1)
            bestScores, bestScoresId = flat_beam_lk.topk(len(active_idx_list), 0, True, True) ## self.size ## active_idx_list ==> for each group!
            scores.scatter_(0, active_idx, bestScores)
            
            # bestScoresId is flattened beam x word array, so calculate which
            # word and beam each score came from
            prev_k = bestScoresId / num_words
            next_ys = bestScoresId - prev_k * num_words

            if self.tt == torch.cuda:
                prev_k1 = torch.arange(0,self.size_g).long().cuda().scatter_(0, active_idx, active_idx[prev_k])        
            else:
                prev_k1 = torch.arange(0,self.size_g).long().scatter_(0, active_idx, active_idx[prev_k])        
                
            next_ys1 = self.tt.LongTensor(self.size_g).fill_(self.pad).scatter_(0, active_idx, next_ys)
        else:
            if self.tt == torch.cuda:
                prev_k1 = torch.arange(0,self.size_g).long().cuda()      
            else:
                prev_k1 = torch.arange(0,self.size_g).long()
            next_ys1 = self.tt.LongTensor(self.size_g).fill_(self.pad)
             
        return scores, prev_k1, next_ys1
        
    
    def advance_diverse(self, word_lk):
        prev_k = self.tt.LongTensor(self.size).zero_()
        next_y = self.tt.LongTensor(self.size).zero_()      

        if len(self.prevKs) > 0:            
            for g in range(self.group):            
                scores, prev_k1, next_ys1 = self.group_advance(word_lk[g*self.size_g:(g+1)*self.size_g], 
                                                               self.scores[g*self.size_g:(g+1)*self.size_g],
                                                               self.score_mask[g*self.size_g:(g+1)*self.size_g],
                                                               self.active_idx[g], ## for each group
                                                               self.active_idx_list[g]) ##for each group
                # appending
                #if g == 0:
                #   pdb.set_trace()
                self.scores[g*self.size_g:(g+1)*self.size_g] = scores
                prev_k[g*self.size_g:(g+1)*self.size_g] = prev_k1 + (g*self.size_g) ## adding the offset
                next_y[g*self.size_g:(g+1)*self.size_g] = next_ys1
        else:
            scores, prev_k1, next_ys1 = self.advance(word_lk) ## select # group, copy-paste!!
            
            for g in range(self.group):
                self.scores[g*self.size_g:(g+1)*self.size_g] = self.tt.FloatTensor(self.size_g).fill_(scores[g])
                prev_k[g*self.size_g:(g+1)*self.size_g] = self.tt.LongTensor(self.size_g).fill_(prev_k1[g])  ## consider the offset
                next_y[g*self.size_g:(g+1)*self.size_g] = self.tt.LongTensor(self.size_g).fill_(next_ys1[g])
                
        self.prevKs.append(prev_k) # trasform prev_k => original index
        self.nextYs.append(next_y)     
        
        if debug:
            print(prev_k)
            print(next_y)
            print(self.scores)
            pdb.set_trace()
        
        # mask
        done = True
        for i in range(self.size):
            if self.nextYs[-1][i]  == self.eos:
                self.doneYs[i] = True
                self.score_mask[i] = 0
                group_idx = int(i/self.size_g)
                beam_idx = i % self.size_g                
                self.active_idx_list[group_idx].remove(beam_idx)
                if debug:
                    #pdb.set_trace()
                    print(i)
                    print(self.active_idx_list[group_idx])
            done *= self.doneYs[i]
        
        for g in range(self.group):
            self.active_idx[g] = self.tt.LongTensor(self.active_idx_list[g])        
                    
        self.done = done
        return self.done    
        
    def advance(self, word_lk):
        """Advance the beam."""
        """The 1st time"""
        
        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0] ## called!
        
        # Avoid generating UNK token
        if not self.gen_unk:
            if beam_lk.dim() == 1:
                beam_lk[self.unk] = -100
            else:
                beam_lk[:, self.unk] = -100           
        
        flat_beam_lk = beam_lk.view(-1)
        
        bestScores, bestScoresId = flat_beam_lk.topk(self.group, 0, True, True)
        scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        next_ys = bestScoresId - prev_k * num_words
        #self.prevKs.append(prev_k)
        #self.nextYs.append(bestScoresId - prev_k * num_words)       
        """
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
            
        return self.done
        """
        return scores, prev_k, next_ys
        
        
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
       
    
    def list2tensor(self, list):
        temp = list[0]
        for i in range(len(list)-2):
            temp = torch.cat(1, temp, list[i+1])
        print(temp)
        
        