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

    