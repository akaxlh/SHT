import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import pairPredict

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        self.uHyper = nn.Parameter(init(t.empty(args.hyperNum, args.latdim)))
        self.iHyper = nn.Parameter(init(t.empty(args.hyperNum, args.latdim)))
    
    def gcnLayer(self, adj, embeds):
        return t.spmm(adj, embeds)
    
    def hgnnLayer(self, embeds, hyper):
        # HGNN can also be seen as learning a transformation in hidden space, with args.hyperNum hidden units (hyperedges)
        return embeds @ (hyper.T @ hyper)# @ (embeds.T @ embeds)
    
    def forward(self, adj):
        embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
        lats = [embeds]
        for i in range(args.gcn_hops):
            temlat = self.gcnLayer(adj, lats[-1])
            lats.append(temlat)
        embeds = sum(lats)
        # this detach helps eliminate the mutual influence between the local GCN and the global HGNN
        hyperUEmbeds = self.hgnnLayer(embeds[:args.user].detach(), self.uHyper)
        hyperIEmbeds = self.hgnnLayer(embeds[args.user:].detach(), self.iHyper)
        return embeds, hyperUEmbeds, hyperIEmbeds

    def pickEdges(self, adj):
        idx = adj._indices()
        rows, cols = idx[0, :], idx[1, :]
        mask = t.logical_and(rows <= args.user, cols > args.user)
        rows, cols = rows[mask], cols[mask]
        edgeSampNum = int(args.edgeSampRate * rows.shape[0])
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        edgeids = t.randint(rows.shape[0], [edgeSampNum])
        pckUsrs, pckItms = rows[edgeids], cols[edgeids] - args.user
        return pckUsrs, pckItms
    
    def pickRandomEdges(self, adj):
        edgeNum = adj._indices().shape[1]
        edgeSampNum = int(args.edgeSampRate * edgeNum)
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        rows = t.randint(args.user, [edgeSampNum])
        cols = t.randint(args.item, [edgeSampNum])
        return rows, cols
    
    def bprLoss(self, uEmbeds, iEmbeds, ancs, poss, negs):
        ancEmbeds = uEmbeds[ancs]
        posEmbeds = iEmbeds[poss]
        negEmbeds = iEmbeds[negs]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - ((scoreDiff).sigmoid() + 1e-6).log().mean()
        return bprLoss
    
    def calcLosses(self, ancs, poss, negs, adj):
        embeds, hyperU, hyperI = self.forward(adj)
        uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]

        bprLoss = self.bprLoss(uEmbeds, iEmbeds, ancs, poss, negs) + self.bprLoss(hyperU, hyperI, ancs, poss, negs)

        if args.ssl1_reg != 0 or args.ssl2_reg != 0:
            # the sample generation can be further generalized as this
            pckUsrs, pckItms = self.pickRandomEdges(adj)
            # we can simply apply a symmetric manner for prediction align, without the cumbersome meta network
            _scores1 = (hyperU[pckUsrs] * hyperI[pckItms]).sum(-1)
            _scores2 = (uEmbeds[pckUsrs] * iEmbeds[pckItms]).sum(-1)
            halfNum = _scores1.shape[0] // 2
            fstScores1 = _scores1[:halfNum]
            scdScores1 = _scores1[halfNum:]
            fstScores2 = _scores2[:halfNum]
            scdScores2 = _scores2[halfNum:]
            scores1 = ((fstScores1 - scdScores1) / args.temp).sigmoid()
            scores2 = ((fstScores2 - scdScores2) / args.temp).sigmoid()
            # prediction alignment in a BPR-like scheme
            sslLoss1 = -(scores2.detach() * (scores1 + 1e-8).log() + (1 - scores2.detach()) * (1 - scores1 + 1e-8).log()).mean() * args.ssl1_reg
            sslLoss2 = -(scores1.detach() * (scores2 + 1e-8).log() + (1 - scores1.detach()) * (1 - scores2 + 1e-8).log()).mean() * args.ssl2_reg
            sslLoss = sslLoss1 + sslLoss2
        else:
            sslLoss = 0
        return bprLoss, sslLoss
    
    def predict(self, adj):
        embeds, hyperU, hyperI = self.forward(adj)
        return hyperU, hyperI