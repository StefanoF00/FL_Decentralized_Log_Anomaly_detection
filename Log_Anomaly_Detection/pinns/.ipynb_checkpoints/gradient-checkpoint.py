import torch

def jacobian(output, input, i=None, j=None, create_graph=True):
    # dimension 0 is the batchsize, so we have to work with all the other dimensions rather
    # than the first one
    # If i == None and j == None, return the full jacobian
    # If, instead i == None and j != None return the j-th colum of the jacobian
    # If i != None and j == None return the i-th row of the jacobian
    # Else return the element (i,j) of the jacobian
    if i == None:
        # compute the full gradient and output it
        grads = []
        for k in range(output.shape[-1]):
            g = torch.zeros_like(output)
            g[..., k] = 1
            # compute the k_th row of the Jacobian
            d_k = torch.autograd.grad(output, input, grad_outputs=g, create_graph=create_graph)[0]
            grads.append(d_k)
        d = torch.stack(grads, dim = 1)
        if j == None:
            return d
        else:
            return d[..., j]
    else:
        g = torch.zeros_like(output)
        g[..., i] = 1
        d = torch.autograd.grad(output, input, grad_outputs=g, create_graph=create_graph)[0]
        if j == None:
            return d
        else:
            return d[..., j]