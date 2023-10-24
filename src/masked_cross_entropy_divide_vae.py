import torch
from torch.nn import functional
from src import config
import torch.nn.functional as F
# from utilis_svae import emd


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))

    return seq_range_expand < seq_length_expand

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def masked_cross_entropy(logits, target, length, logits_noGrad = None, target_noGrad = None, length_noGrad = None, temperature = None, mean = None, logvar = None, problem_output_sample=None ,problem_output_sample_noGrad = None , is_loss_no_mask=False):

    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits(all_node_outputs): A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
            

        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    def compute_kl_loss(all_node_outputs_old, pad_mask, logits_noGrad = None, features2_mask = None):
        
        p_tec, q_tec = functional.softmax(all_node_outputs_old, dim=-1), functional.softmax(logits_noGrad, dim=-1)
        p, q = functional.log_softmax(all_node_outputs_old, dim=-1), functional.log_softmax(logits_noGrad, dim=-1)

        # all_node_outputs_tec = functional.softmax(all_node_outputs_old, dim=-1)
        # # print("all_node_outputs_tec:{}".format(all_node_outputs_tec))
        # all_node_outputs = functional.log_softmax(all_node_outputs_old, dim=-1)
        # # print("all_node_outputs:{}".format(all_node_outputs))

        # p, q = torch.split(all_node_outputs, all_node_outputs.size(0)//2, dim=0)

        # p_tec, q_tec = torch.split(all_node_outputs_tec, all_node_outputs_tec.size(0)//2, dim=0)
        
        p_mask, q_mask = pad_mask, features2_mask#torch.split(pad_mask, pad_mask.size(0)//2, dim=0)
        # p_tec, q_tec = p_tec.masked_fill_(p_mask, 0.), q_tec.masked_fill_(q_mask, 0.)

        # print("p_tec shape:{} \n q_tec shape:{}".format(p_tec.shape, q_tec.shape))
        # print("p shape:{} \n q shape:{}".format(p.shape, q.shape))
        p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
        q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
        # print("p_loss1:{} \n q_loss1:{}".format(p_loss, q_loss))
        # print("p_loss shape:{} \n q_loss shape:{}".format(p_loss.shape, q_loss.shape))

        if pad_mask is not None:

            # p_mask = torch.stack((p_mask, p_mask, p_mask, p_mask, p_mask, p_mask, p_mask,), 2)
            # q_mask = torch.stack((q_mask, q_mask, q_mask), 2)
            # print("p_loss shape:{}  p_mask shape:{}".format(p_loss.shape, p_mask.shape))
            # print("p_mask:{}".format(p_mask))
            if is_loss_no_mask ==True:
                p_loss = p_loss.sum(2)#.masked_fill_(~p_mask, 0.)
                q_loss = q_loss.sum(2)#.masked_fill_(~q_mask, 0.)
            else:
                p_loss = p_loss.sum(2).masked_fill_(~p_mask, 0.)
                q_loss = q_loss.sum(2).masked_fill_(~q_mask, 0.)

            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def compute_wasserstein_loss(all_node_outputs_old ,pad_mask = None):
        all_node_outputs_tec = functional.softmax(all_node_outputs_old, dim=-1)
        p, q = torch.split(all_node_outputs_tec, all_node_outputs_tec.size(0)//2, dim=0)
        p_mask, q_mask = torch.split(pad_mask, pad_mask.size(0)//2, dim=0)
        
        # print("p_mask shape:{}".format(p_mask.shape))
        # print("p shape:{}".format(p.shape))
        # print("p[0][0].masked_fill:{}".format( p.masked_fill(~p_mask.unsqueeze(2).expand(p.size(0), p.size(1), p.size(-1)), 0)[0][0] ))

        p = p.masked_fill(~p_mask.unsqueeze(2).expand(p.size(0), p.size(1), p.size(-1)), 0).sum(dim=1) 
        q = q.masked_fill(~q_mask.unsqueeze(2).expand(q.size(0), q.size(1), q.size(-1)), 0).sum(dim=1) 

        p_mean = p  / p_mask.float().sum(dim=1).unsqueeze(-1)
        q_mean = q  / q_mask.float().sum(dim=1).unsqueeze(-1)

        wasserstein_loss = imq_kernel(p_mean, q_mean, h_dim=all_node_outputs_tec.size(-1))

        return wasserstein_loss

    def nt_xent(features1, features2 , features1_mask = None, features2_mask = None, t=0.5):
        #print("features2 shape:{} features2_mask shape:{}".format(features2.shape, features2_mask.shape))
        #print("features1_mask.unsqueeze(2).expand(features1.size(0), features1.size(1), features1.size(-1)) shape:{}".format(features1_mask.unsqueeze(2).expand(features1.size(0), features1.size(1), features1.size(-1)).shape))

        p = features1.masked_fill(~features1_mask.unsqueeze(2).expand(features1.size(0), features1.size(1), features1.size(-1)), 0).sum(dim=1) 

        q = features2.masked_fill(~features2_mask.unsqueeze(2).expand(features2.size(0), features2.size(1), features2.size(-1)), 0).sum(dim=1) 

        p_mean = p  / features1_mask.float().sum(dim=1).unsqueeze(-1)
        q_mean = q  / features2_mask.float().sum(dim=1).unsqueeze(-1)
        
        batch_size = features1.shape[0]
        out_1 = F.normalize(p_mean, dim=-1)
        out_2 = F.normalize(q_mean, dim=-1)

        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        # print("temperature is {}".format(t))
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

        mask = get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        Ng = neg.sum(dim=-1)

        # contrastive loss

        loss = (- torch.log(pos / (pos + Ng)))

        return loss.mean()

    def cosine_sim(logits, logits_1):
        return torch.ones(logits.size(0)).cuda() + torch.cosine_similarity(logits, logits_1, dim=1).cuda()

    def cosine_loss(logits, logits_1, length, mask = None):
        loss_total = []
        for predict,label in zip(logits.split(1,dim=1),logits_1.split(1,dim=1)):
            predict = predict.squeeze()
            label = label.squeeze()
            loss_t = cosine_sim(predict,label)
            loss_total.append(loss_t)
        
        if mask !=None:
            loss_total = torch.stack(loss_total,dim=0).transpose(1,0)* mask.float()
        else:
            loss_total = torch.stack(loss_total,dim=0).transpose(1,0)
        #loss_total = loss_total.sum(dim=1)
        loss_total = loss_total.sum() / length.float().sum()
        return loss_total

    def kl_loss_diag(mean, logvar, logvar_prior=None, average=True):
        if logvar_prior is None:
            result = -0.5 * torch.sum(logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1, 1)
        else:
            logvar_prior = logvar_prior.unsqueeze(0).expand_as(logvar)        
            result =  -0.5 * torch.sum(logvar - logvar_prior - torch.pow(mean, 2) / logvar_prior.exp() -
                                    torch.exp(logvar - logvar_prior) + 1, 1)
        if average:
            return result.mean()
        else:
            return result

    

    
    
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # print("losses_flat:{}".format(losses_flat))

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    # print("length1:{}".format(length))
    

    # print("losses: mask shape:{}".format(losses))
    
    

    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum() 

    if config.is_RDrop and config.is_train_kl:
       if config.RDloss == "kl_loss":
           kl_loss = compute_kl_loss(logits, mask, logits_noGrad = logits_noGrad, features2_mask = mask)
       elif config.RDloss == "wasserstein_loss":
           kl_loss = compute_wasserstein_loss(logits, mask)
       elif config.RDloss == "cosine_loss":
           kl_loss = cosine_loss(logits, logits_noGrad, length, mask = mask)
       else:
        #    print("length:{} \n length_noGrad shape:{}".format(length, length_noGrad))
        #    print("length2:{}".format(length_noGrad))
           #mask_noGrad = sequence_mask(sequence_length=torch.LongTensor(length_noGrad).cuda(), max_len=target_noGrad.size(1))
           kl_loss = nt_xent(logits, logits_noGrad, features1_mask = mask, features2_mask = mask, t=temperature)
        #    print("wasserstein_loss:{}".format(kl_loss))

       loss   += config.contra_weight * kl_loss
       
       if config.is_vae:
          
          vae_kl = kl_loss_diag(mean, logvar)

          #print("vae_kl:{}".format(vae_kl))
          loss += vae_kl
          return loss, kl_loss, vae_kl
       else:
          return loss, kl_loss
    else:
       return loss, 0





def masked_cross_entropy_dice(logits, target, length, logits_noGrad = None, target_noGrad = None, length_noGrad = None, temperature = None, mean = None, logvar = None, problem_output_sample=None ,problem_output_sample_noGrad = None , is_loss_no_mask=False,
                              all_node_outputs_embeddings = None):

    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits(all_node_outputs): A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
            

        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    def compute_kl_loss(all_node_outputs_old, pad_mask, logits_noGrad = None, features2_mask = None):
        
        p_tec, q_tec = functional.softmax(all_node_outputs_old, dim=-1), functional.softmax(logits_noGrad, dim=-1)
        p, q = functional.log_softmax(all_node_outputs_old, dim=-1), functional.log_softmax(logits_noGrad, dim=-1)

        # all_node_outputs_tec = functional.softmax(all_node_outputs_old, dim=-1)
        # # print("all_node_outputs_tec:{}".format(all_node_outputs_tec))
        # all_node_outputs = functional.log_softmax(all_node_outputs_old, dim=-1)
        # # print("all_node_outputs:{}".format(all_node_outputs))

        # p, q = torch.split(all_node_outputs, all_node_outputs.size(0)//2, dim=0)

        # p_tec, q_tec = torch.split(all_node_outputs_tec, all_node_outputs_tec.size(0)//2, dim=0)
        
        p_mask, q_mask = pad_mask, features2_mask#torch.split(pad_mask, pad_mask.size(0)//2, dim=0)
        # p_tec, q_tec = p_tec.masked_fill_(p_mask, 0.), q_tec.masked_fill_(q_mask, 0.)

        # print("p_tec shape:{} \n q_tec shape:{}".format(p_tec.shape, q_tec.shape))
        #print("p shape:{} \n q shape:{}".format(p.shape, q.shape))

        p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
        q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
        # print("p_loss1:{} \n q_loss1:{}".format(p_loss, q_loss))
        # print("p_loss shape:{} \n q_loss shape:{}".format(p_loss.shape, q_loss.shape))

        if pad_mask is not None:

            # p_mask = torch.stack((p_mask, p_mask, p_mask, p_mask, p_mask, p_mask, p_mask,), 2)
            # q_mask = torch.stack((q_mask, q_mask, q_mask), 2)
            # print("p_loss shape:{}  p_mask shape:{}".format(p_loss.shape, p_mask.shape))
            # print("p_mask:{}".format(p_mask))
            if is_loss_no_mask ==True:
                p_loss = p_loss.sum(2)#.masked_fill_(~p_mask, 0.)
                q_loss = q_loss.sum(2)#.masked_fill_(~q_mask, 0.)
            else:
                p_loss = p_loss.sum(2).masked_fill_(~p_mask, 0.)
                q_loss = q_loss.sum(2).masked_fill_(~q_mask, 0.)

            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def compute_CEB_loss(all_node_outputs_old, logits_noGrad = None):
        
        p_tec, q_tec = functional.softmax(all_node_outputs_old, dim=-1), functional.softmax(logits_noGrad, dim=-1)
        p, q = functional.log_softmax(all_node_outputs_old, dim=-1), functional.log_softmax(logits_noGrad, dim=-1)

        # all_node_outputs_tec = functional.softmax(all_node_outputs_old, dim=-1)
        # # print("all_node_outputs_tec:{}".format(all_node_outputs_tec))
        # all_node_outputs = functional.log_softmax(all_node_outputs_old, dim=-1)
        # # print("all_node_outputs:{}".format(all_node_outputs))

        # p, q = torch.split(all_node_outputs, all_node_outputs.size(0)//2, dim=0)

        # p_tec, q_tec = torch.split(all_node_outputs_tec, all_node_outputs_tec.size(0)//2, dim=0)
        
        # p_mask, q_mask = pad_mask, features2_mask#torch.split(pad_mask, pad_mask.size(0)//2, dim=0)
        # p_tec, q_tec = p_tec.masked_fill_(p_mask, 0.), q_tec.masked_fill_(q_mask, 0.)

        # print("p_tec shape:{} \n q_tec shape:{}".format(p_tec.shape, q_tec.shape))
        # print("p shape:{} \n q shape:{}".format(p.shape, q.shape))
        p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
        q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
        # print("p_loss1:{} \n q_loss1:{}".format(p_loss, q_loss))
        # print("p_loss shape:{} \n q_loss shape:{}".format(p_loss.shape, q_loss.shape))

       
        # p_loss = p_loss.sum(2)#.masked_fill_(~p_mask, 0.)
        # q_loss = q_loss.sum(2)#.masked_fill_(~q_mask, 0.)
        # print("p_loss shape:{}".format(p_loss.shape))   
        
        p_loss = p_loss.sum(1)
        q_loss = q_loss.sum(1)

        loss = (p_loss + q_loss) / 2
        #print("loss shape:{}".format(loss.shape))   
        return loss.mean()

    def compute_CEB_loss_2(all_node_outputs_old, logits_noGrad = None):
        
        p_tec, q_tec = functional.softmax(all_node_outputs_old, dim=-1), functional.softmax(logits_noGrad, dim=-1)
        p, q = functional.log_softmax(all_node_outputs_old, dim=-1), functional.log_softmax(logits_noGrad, dim=-1)

        # all_node_outputs_tec = functional.softmax(all_node_outputs_old, dim=-1)
        # # print("all_node_outputs_tec:{}".format(all_node_outputs_tec))
        # all_node_outputs = functional.log_softmax(all_node_outputs_old, dim=-1)
        # # print("all_node_outputs:{}".format(all_node_outputs))

        # p, q = torch.split(all_node_outputs, all_node_outputs.size(0)//2, dim=0)

        # p_tec, q_tec = torch.split(all_node_outputs_tec, all_node_outputs_tec.size(0)//2, dim=0)
        
        # p_mask, q_mask = pad_mask, features2_mask#torch.split(pad_mask, pad_mask.size(0)//2, dim=0)
        # p_tec, q_tec = p_tec.masked_fill_(p_mask, 0.), q_tec.masked_fill_(q_mask, 0.)

        # print("p_tec shape:{} \n q_tec shape:{}".format(p_tec.shape, q_tec.shape))
        # print("p shape:{} \n q shape:{}".format(p.shape, q.shape))
        p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
        q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
        # print("p_loss1:{} \n q_loss1:{}".format(p_loss, q_loss))
        # print("p_loss shape:{} \n q_loss shape:{}".format(p_loss.shape, q_loss.shape))

       
        # p_loss = p_loss.sum(2)#.masked_fill_(~p_mask, 0.)
        # q_loss = q_loss.sum(2)#.masked_fill_(~q_mask, 0.)
        # print("p_loss shape:{}".format(p_loss.shape))   
        
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        #print("loss shape:{}".format(loss.shape))   
        return loss

    def compute_wasserstein_loss(all_node_outputs_old ,pad_mask = None):
        all_node_outputs_tec = functional.softmax(all_node_outputs_old, dim=-1)
        p, q = torch.split(all_node_outputs_tec, all_node_outputs_tec.size(0)//2, dim=0)
        p_mask, q_mask = torch.split(pad_mask, pad_mask.size(0)//2, dim=0)
        
        # print("p_mask shape:{}".format(p_mask.shape))
        # print("p shape:{}".format(p.shape))
        # print("p[0][0].masked_fill:{}".format( p.masked_fill(~p_mask.unsqueeze(2).expand(p.size(0), p.size(1), p.size(-1)), 0)[0][0] ))

        p = p.masked_fill(~p_mask.unsqueeze(2).expand(p.size(0), p.size(1), p.size(-1)), 0).sum(dim=1) 
        q = q.masked_fill(~q_mask.unsqueeze(2).expand(q.size(0), q.size(1), q.size(-1)), 0).sum(dim=1) 

        p_mean = p  / p_mask.float().sum(dim=1).unsqueeze(-1)
        q_mean = q  / q_mask.float().sum(dim=1).unsqueeze(-1)

        wasserstein_loss = imq_kernel(p_mean, q_mean, h_dim=all_node_outputs_tec.size(-1))

        return wasserstein_loss

    def nt_xent(features1, features2 , features1_mask = None, features2_mask = None, t=0.5):
        #print("features2 shape:{} features2_mask shape:{}".format(features2.shape, features2_mask.shape))
        #print("features1_mask.unsqueeze(2).expand(features1.size(0), features1.size(1), features1.size(-1)) shape:{}".format(features1_mask.unsqueeze(2).expand(features1.size(0), features1.size(1), features1.size(-1)).shape))

        p = features1.masked_fill(~features1_mask.unsqueeze(2).expand(features1.size(0), features1.size(1), features1.size(-1)), 0).sum(dim=1) 

        q = features2.masked_fill(~features2_mask.unsqueeze(2).expand(features2.size(0), features2.size(1), features2.size(-1)), 0).sum(dim=1) 

        p_mean = p  / features1_mask.float().sum(dim=1).unsqueeze(-1)
        q_mean = q  / features2_mask.float().sum(dim=1).unsqueeze(-1)
        
        batch_size = features1.shape[0]
        out_1 = F.normalize(p_mean, dim=-1)
        out_2 = F.normalize(q_mean, dim=-1)

        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        # print("temperature is {}".format(t))
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

        mask = get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        Ng = neg.sum(dim=-1)

        # contrastive loss

        loss = (- torch.log(pos / (pos + Ng)))

        return loss.mean()

    def cosine_sim(logits, logits_1):
        return torch.ones(logits.size(0)).cuda() + torch.cosine_similarity(logits, logits_1, dim=1).cuda()

    def cosine_loss(logits, logits_1, length, mask = None):
        loss_total = []
        for predict,label in zip(logits.split(1,dim=1),logits_1.split(1,dim=1)):
            predict = predict.squeeze()
            label = label.squeeze()
            loss_t = cosine_sim(predict,label)
            loss_total.append(loss_t)
        
        if mask !=None:
            loss_total = torch.stack(loss_total,dim=0).transpose(1,0)* mask.float()
        else:
            loss_total = torch.stack(loss_total,dim=0).transpose(1,0)
        #loss_total = loss_total.sum(dim=1)
        loss_total = loss_total.sum() / length.float().sum()
        return loss_total

    def kl_loss_diag(mean, logvar, logvar_prior=None, average=True):
        if logvar_prior is None:
            result = -0.5 * torch.sum(logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1, 1)
        else:
            logvar_prior = logvar_prior.unsqueeze(0).expand_as(logvar)        
            result =  -0.5 * torch.sum(logvar - logvar_prior - torch.pow(mean, 2) / logvar_prior.exp() -
                                    torch.exp(logvar - logvar_prior) + 1, 1)
        if average:
            return result.mean()
        else:
            return result

    

    
    
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # print("losses_flat:{}".format(losses_flat))

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    # print("length1:{}".format(length))
    

    # print("losses: mask shape:{}".format(losses))
    
    

    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum() 
    
    #print("logits shape:{}".format(all_node_outputs_embeddings.shape))
    mean_all_node_outputs_embeddings = torch.mean(all_node_outputs_embeddings, dim = 1)
    # print("mean_logits shape:{}".format(mean_all_node_outputs_embeddings.shape))
    # print("problem_output_sample shape:{}".format(problem_output_sample.shape))

    CEB_loss = compute_CEB_loss(mean_all_node_outputs_embeddings.squeeze(1) ,  logits_noGrad = problem_output_sample)

    #print("CEB_loss: {}".format(CEB_loss))

    if config.is_RDrop and config.is_train_kl:
       if config.RDloss == "kl_loss":
           kl_loss = compute_kl_loss(logits, mask, logits_noGrad = logits_noGrad, features2_mask = mask)
       elif config.RDloss == "wasserstein_loss":
           kl_loss = compute_wasserstein_loss(logits, mask)
       elif config.RDloss == "cosine_loss":
           kl_loss = cosine_loss(logits, logits_noGrad, length, mask = mask)
       else:
        #    print("length:{} \n length_noGrad shape:{}".format(length, length_noGrad))
        #    print("length2:{}".format(length_noGrad))
           #mask_noGrad = sequence_mask(sequence_length=torch.LongTensor(length_noGrad).cuda(), max_len=target_noGrad.size(1))
           kl_loss = nt_xent(logits, logits_noGrad, features1_mask = mask, features2_mask = mask, t=temperature)
        #    print("wasserstein_loss:{}".format(kl_loss))

       loss   += config.contra_weight * kl_loss
       loss   += config.contra_weight * CEB_loss
       if config.is_vae:
        #   if config.is_vmf:
               
        #        sub_batch_size = 16
        #        z_x_2 = problem_output_sample.rsample(sub_batch_size).permute(1,0,2)
        #        z_attr_2 = problem_output_sample_noGrad.rsample(sub_batch_size).permute(1,0,2)
        #        sinkhorn_loss = emd.SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
        #        dist, P, C = sinkhorn_loss(z_x_2, z_attr_2)
        #        vae_kl = dist.mean()
        #   else:
          vae_kl = kl_loss_diag(mean, logvar)

               
          loss += vae_kl
          return loss, kl_loss, vae_kl, CEB_loss
       else:
          return loss, kl_loss
    else:
       return loss, 0
    


def masked_cross_entropy_without_logit(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))

    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.log(logits_flat + 1e-12)

    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())

    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss



def masked_cross_entropy_comp_noVAE(logits, target, length, logits_noGrad = None, target_noGrad = None, length_noGrad = None, temperature = None, mean = None, logvar = None):

    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits(all_node_outputs): A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
            

        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
   

    def kl_loss_diag(mean, logvar, logvar_prior=None, average=True):
        if logvar_prior is None:
            result = -0.5 * torch.sum(logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1, 1)
        else:
            logvar_prior = logvar_prior.unsqueeze(0).expand_as(logvar)        
            result =  -0.5 * torch.sum(logvar - logvar_prior - torch.pow(mean, 2) / logvar_prior.exp() -
                                    torch.exp(logvar - logvar_prior) + 1, 1)
        if average:
            return result.mean()
        else:
            return result
    
    
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # print("losses_flat:{}".format(losses_flat))

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    # print("length1:{}".format(length))
    

    # print("losses: mask shape:{}".format(losses))
    
    

    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum() 

   

       
    return loss






