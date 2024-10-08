import math
import numpy as np
import torch


def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n]).to("cuda")
    I = torch.eye(n).to("cuda")
    H = I - unit / n

    return torch.mm(torch.mm(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering


def rbf(X, sigma=None):
    GX = torch.mm(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = torch.mm(X, X.T)
    L_Y = torch.mm(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def cka_logits(feat, proto):
    # equivalent to linear_CKA, batch computation
    # feat: [b, c, h*w]
    # proto: [num_classes, c, hp*wp]
    def centering(feat):
        assert len(feat.shape) == 3
        return feat - torch.mean(feat, dim=1, keepdims=True)
    
    def cka(va, vb):
        return torch.norm(torch.matmul(va.t(), vb)) ** 2 / (torch.norm(torch.matmul(va.t(), va)) * torch.norm(torch.matmul(vb.t(), vb)))
    
    proto = centering(proto)
    feat = centering(feat)

    ### equivalent implementation ###
    proto = proto.unsqueeze(0) # [1, num_classes, c, hp*wp]
    feat = feat.unsqueeze(1) # [b, 1, c, h*w]

    cross_norm = torch.norm(torch.matmul(feat.permute(0, 1, 3, 2), proto), dim=[2,3]) ** 2 # [b, num_classes]
    feat_norm = torch.norm(torch.matmul(feat.permute(0, 1, 3, 2), feat), dim=[2,3]) # [b, 1]
    proto_norm = torch.norm(torch.matmul(proto.permute(0, 1, 3, 2), proto), dim=[2,3]) # [1, num_classes]

    logits = cross_norm / (feat_norm * proto_norm) # [b, num_classes]

    return logits

if __name__=='__main__':
    feat = torch.randn(24,128,25).to("cuda")
    proto = torch.randn(100, 128, 10).to("cuda")

    logits = cka_logits(feat, proto)

    print('RBF Kernel CKA, between X and Y: {}{}'.format(logits, logits.shape))