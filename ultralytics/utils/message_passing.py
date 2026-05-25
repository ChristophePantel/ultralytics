# Disjunction: specialisation / decomposition
def disjunction_message(self, pred_scores, targets_from_source):
    batch_size, anchor_point_size, class_number = pred_scores.shape
    sources = list(targets_from_source )
    source_number = len(sources)
    predicate_s = torch.zeros(source_number)
    for idx_s, s in enumerate(sources):
        targets_from_s = list(targets_from_source[s])
        # p_c(o)
        source_scores = pred_scores[:, :, s:s+1]
        # p_d(o)
        indexes = torch.tensor(targets_from_s,device=pred_scores.device)
        target_scores = pred_scores.index_select( 2, indexes)
        # max(p_d(o))
        target_scores_max = target_scores.max()
        # p_s(o) - p_s(o) * max(p_t(o))
        predicate_s_o = 1 - source_scores + source_scores * target_scores_max
        # TODO: Is it meaningful ?
        # p-mean on O
        # predicate_s[idx_s] = torch.pow(torch.pow(predicate_s_o,power).mean(dim=(0,1)),1/power)
    # mean for s in sources
    # result = torch.mean(predicate_s)
    result = predicate_s_o
    return result

    # One of the targets for a given valid source must be valid : forall s in S, forall o in O, p_s(o) -> \/_{t in T(s)} p_t(o)
    # origin_indexes : class indexes for the domain
    # targets_from_source : class indexes for the co-domain
    def disjunction_message(self, pred_scores, targets_from_source, power=3.0):
        # O = batch_size x anchor_point_size
        batch_size, anchor_point_size, class_number = pred_scores.shape
        # S
        sources = list(targets_from_source )
        source_number = len(sources)
        # indexed by s in S
        predicate_s = torch.zeros(source_number)
        # for each s in S
        for idx_s, s in enumerate(sources):
            # t in T(s)
            targets_from_s = list(targets_from_source[s])
            # p_s(o): indexed by o for a given s
            source_scores = pred_scores[:, :, s]
            # T(s)
            indexes = torch.tensor(targets_from_s,device=pred_scores.device)
            # p_t(o): indexed by o in O and by t in T(s) for a given s in S
            target_scores = pred_scores.index_select( 2, indexes)
            # max(p_t(o)) for t in T(s) 
            target_scores_max = torch.amax(target_scores,2)
            # p_s(o) - p_s(o) * max(p_t(o)): indexed by s and o
            predicate_s_o = source_scores - source_scores * target_scores_max
            # p-mean for o in O for a given s in S, o (dim 1) is defined for each image in a batch (dim 0)
            predicate_s[idx_s] = torch.pow(predicate_s_o,power).mean(dim=(0,1))
        # p-mean for s in S
        result = torch.pow(torch.mean(predicate_s),1/power)
        return result

# Conjunction: generalisation / composition
def conjunction_message(self, pred_scores, targets_from_source):
    batch_size, anchor_point_size, class_number = pred_scores.shape
    sources = list(targets_from_source )
    source_number = len(sources)
    predicate_s = torch.zeros(source_number)
    for idx_s, s in enumerate(sources):
        # p_s(o)
        source_scores = pred_scores[:,:,s:s+1]
        targets_from_s = list(targets_from_source[s])
        indexes = torch.tensor(targets_from_s,device=pred_scores.device)
        s_targets_number = len(targets_from_s)
        # p_t(o)
        target_scores_t = pred_scores.index_select( 2, indexes)
        # p_s(o) - p_s(o) * p_t(o)
        predicate_s_t_o = 1 - source_scores + source_scores * target_scores_t
        # TODO: Is it meaningful ?
        # p-mean on O
        # predicate_s_t = torch.pow(torch.pow(predicate_s_t_o,power).mean(dim=(0,1)),1/power)
        # mean on A
        # predicate_s[idx_s] = torch.mean(predicate_s_t)
        # mean on C \ R
        # result = torch.mean(predicate_s)
        result = predicate_s_t_o
    return result

# Exclusion: generalisation / composition
def exclusion_message(self, pred_scores, targets_from_source):
    batch_size, anchor_point_size, class_number = pred_scores.shape
    sources = list(targets_from_source )
    source_number = len(sources)
    predicate_s = torch.zeros(source_number)
    for idx_s, s in enumerate(sources):
        # p_s(o)
        source_scores = pred_scores[:,:,s:s+1]
        targets_from_s = list(targets_from_source[s])
        s_targets_number = len(targets_from_s)
        predicate_s_t = torch.zeros(s_targets_number)
        if s_targets_number > 1:
            for idx_t, t in enumerate(targets_from_s):
                targets_except_t = targets_from_s.copy()
                targets_except_t.remove(t)
                # p_t(o)
                target_scores_t = pred_scores[:,:, t:t+1]
                # p_e(o)
                indexes = torch.tensor(targets_except_t,device=pred_scores.device)
                target_scores_e = pred_scores.index_select( 2, indexes)
                # p_t(o) * p_e(o)
                predicate_s_t_e_o = target_scores_t * target_scores_e
                # TODO: Is it meaningful ?
                # p-mean on O
                # predicate_s_t_e = torch.pow(torch.pow(predicate_s_t_e_o,power).mean(dim=(0,1)),1/power)
                # mean on e in T(s) \ { t }
                # predicate_s_t[idx_t] = torch.mean(predicate_s_t_e)
        # mean on t in T(s)
        # predicate_s[idx_s] = torch.mean(predicate_s_t)
    # mean on s in S
    # result = torch.mean(predicate_s)
    # TODO: a mean is missing on
    result = predicate_s_t_e_o
    return result

def message_passing(self, pred_scores, targets_from_source):
    S_message = specialisation_message(self, pred_scores, targets_from_source)
    G_message = generalisation_message(self, pred_scores, targets_from_source)
    E_message = exclusion_message(self, pred_scores, targets_from_source)
    target_scores_max = target_scores.max()
    pred_scores = pred_scores + G_message * TODO + S_message * target_scores_max + E_message * targets_from_source 

    # Obtenir le top-scoring path

    