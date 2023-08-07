from torch import nn
import torch
import numpy as np

def alpha_variance_loss(actions, advantages, adv_grads, model, old_mu, old_sigma, curr_mu, curr_sigma, alpha):
    
    '''
    Compute variance of (alpha-policy) estimator.
    '''

    adv_grads_norm = torch.pow(torch.norm(adv_grads, p=2, dim=1), 2.0)
    
    p_actions = actions + (adv_grads * alpha)
    p_advantages = advantages + (adv_grads_norm * alpha)

    # compute probabilities of [p_actions];

    old_logstd = torch.log(old_sigma)
    curr_logstd = torch.log(curr_sigma)

    old_neglogp = model.neglogp(p_actions, old_mu, old_sigma, old_logstd)
    curr_neglogp = model.neglogp(p_actions, curr_mu, curr_sigma, curr_logstd)

    old_neglogp = torch.squeeze(old_neglogp)
    curr_neglogp = torch.squeeze(curr_neglogp)

    ratio = torch.exp(old_neglogp - curr_neglogp)
    mean_terms = p_advantages * ratio

    a_est_mean = torch.mean(mean_terms, dim=0, keepdim=True)
                
    var_terms = torch.pow(mean_terms - a_est_mean, 2.0)
    a_est_var = torch.mean(var_terms, dim=0).squeeze()

    return a_est_var

def alpha_policy_loss(actions, advantages, adv_grads, model, old_mu, old_sigma, curr_mu, curr_sigma, alpha):
    
    '''
    Compute variance of (alpha-policy) estimator.
    '''

    adv_grads_norm = torch.pow(torch.norm(adv_grads, p=2, dim=1), 2.0)
    
    p_actions = actions + (adv_grads * alpha)
    p_advantages = advantages + (adv_grads_norm * alpha)

    # compute probabilities of [p_actions];

    old_logstd = torch.log(old_sigma)
    curr_logstd = torch.log(curr_sigma)

    old_neglogp = model.neglogp(actions, old_mu, old_sigma, old_logstd)
    curr_neglogp = model.neglogp(p_actions, curr_mu, curr_sigma, curr_logstd)

    old_neglogp = torch.squeeze(old_neglogp)
    curr_neglogp = torch.squeeze(curr_neglogp)

    ratio = torch.exp(old_neglogp - curr_neglogp)
    mean_terms = p_advantages * ratio

    a_est_mean = torch.mean(mean_terms, dim=0, keepdim=True)
                
    var_terms = torch.pow(mean_terms - a_est_mean, 2.0)
    a_est_var = torch.mean(var_terms, dim=0).squeeze()

    return a_est_mean, a_est_var


def alpha_policy_correspondence_loss(actions, advantages, adv_grads, model, old_mu, old_sigma, curr_mu, curr_sigma, alpha):
    
    adv_grads_norm = torch.pow(torch.norm(adv_grads, p=2, dim=1), 2.0)
    
    p_actions = actions + (adv_grads * alpha)
    p_advantages = advantages + (adv_grads_norm * alpha)

    # compute probabilities of [p_actions];

    old_logstd = torch.log(old_sigma)
    curr_logstd = torch.log(curr_sigma)

    old_neglogp = model.neglogp(actions, old_mu, old_sigma, old_logstd)     
    curr_neglogp = model.neglogp(p_actions, curr_mu, curr_sigma, curr_logstd)

    old_neglogp = torch.squeeze(old_neglogp)
    curr_neglogp = torch.squeeze(curr_neglogp)

    ratio = old_neglogp - curr_neglogp      # these two have to be equal;
    loss = torch.norm(ratio, p=2)

    return loss

# WARNING: Not for alpha-policy based grad-ppo.
def alpha_actor_loss(old_action_log_probs_batch, action_log_probs, advantage, is_ppo, curr_e_clip, initial_ratio):

    if is_ppo:
        ratio = old_action_log_probs_batch - action_log_probs
        ratio = torch.clamp(ratio, max=16.0)        # prevent ratio becoming [inf];
        ratio = torch.exp(ratio)
        
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 
                                initial_ratio * (1.0 - curr_e_clip),
                                initial_ratio * (1.0 + curr_e_clip))
        a_loss = torch.max(-surr1, -surr2)

        # for stat;
        num_worse_indices = len(torch.where(surr1 < surr2)[0])
        num_total_indices = len(surr1)
        worse_ratio = num_worse_indices / num_total_indices
    else:
        a_loss = (action_log_probs * advantage)
        worse_ratio = -1.0
    
    return a_loss, worse_ratio

# original actor loss;
def actor_loss(old_action_log_probs_batch, action_log_probs, advantage, is_ppo, curr_e_clip):
    if is_ppo:
        ratio = old_action_log_probs_batch - action_log_probs
        ratio = torch.clamp(ratio, max=64.0)        # prevent ratio becoming [inf];
        ratio = torch.exp(ratio)
        
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip,
                                1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
    else:
        a_loss = (action_log_probs * advantage)
    
    return a_loss

# actor loss for alpha-policy based grad ppo;
# @ old_action_log_probs_batch0: neg action log probs before alpha-policy update
# @ old_action_log_probs_batch1: neg action log probs after alpha-policy update
def  actor_loss_alpha(old_action_log_probs_batch0, 
                     old_action_log_probs_batch1,
                     action_log_probs, 
                     advantage, 
                     is_ppo, 
                     curr_e_clip):
    if is_ppo:
        t_ratio = old_action_log_probs_batch0 - old_action_log_probs_batch1
        if torch.any(torch.abs(t_ratio) > 4.):
            # ratio can be numerically unstable, just use original ppo;
            # but use policy after RP update as importance sampling distribution;
            ratio = old_action_log_probs_batch1 - action_log_probs
        else:
            t_ratio = torch.exp(t_ratio)
            tmp0 = torch.log(t_ratio + 1.)
            tmp1 = tmp0 - old_action_log_probs_batch0
            action_log_probs_batch_mid = np.log(2.) - tmp1
            
            ratio = action_log_probs_batch_mid - action_log_probs
            
        ratio = torch.clamp(ratio, min=-16., max=16.)        # prevent ratio becoming [inf];
        ratio = torch.exp(ratio)

        # if torch.any(torch.isnan(ratio)) or torch.any(torch.isinf(ratio)):
        #     print("Ratio is NAN or INF")
        #     route0 = torch.any(torch.abs(t_ratio) > 4.)
        #     print(f"Route0: {route0}")
        #     print(torch.any(torch.isnan(old_action_log_probs_batch1)) or torch.any(torch.isinf(old_action_log_probs_batch1)))
        #     print(torch.any(torch.isnan(old_action_log_probs_batch0)) or torch.any(torch.isinf(old_action_log_probs_batch0)))
        #     print(torch.any(torch.isnan(action_log_probs)) or torch.any(torch.isinf(action_log_probs)))
        #     print(torch.any(torch.isnan(t_ratio)) or torch.any(torch.isinf(t_ratio)))
        #     exit(-1)

        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip,
                                1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
    else:
        raise NotImplementedError()
    
    return a_loss