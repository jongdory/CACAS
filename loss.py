import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import SimpleITK as sitk

from einops import rearrange, reduce, repeat


def dice_coef_metric(probabilities: torch.Tensor,
                     truth: torch.Tensor,
                     eps: float = 1e-8,
                     maskmats: torch.Tensor = None) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    num = truth.size(0)
    probabilities = probabilities.to("cuda")
    truth = truth.to("cuda")
    maskmats = maskmats.to("cuda")

    probabilities = probabilities.view(num, -1)
    truth = truth.view(num, -1)
    maskmats = maskmats.view(num,-1)
    truth = torch.mul(truth, maskmats)
    probabilities = torch.mul(probabilities, maskmats)
    
    intersection = 2.0 * (probabilities * truth).sum()
    union = probabilities.sum() + truth.sum()
    dice = (intersection + eps) / (union + eps)

    return dice.cpu().numpy()


def binary_cross_entropy_metric(logits: torch.Tensor,
               truth: torch.Tensor,
               maskmats: torch.Tensor = None) -> np.ndarray:

    assert(logits.shape == truth.shape)

    logits = logits.to("cuda")
    truth = truth.to("cuda")
    maskmats = maskmats.to("cuda")

    bce = nn.BCEWithLogitsLoss(pos_weight=maskmats)(logits, truth)

    return bce.cpu().numpy()


def center_distance_metric(probabilities: torch.Tensor,
               eps: float = 1e-8,
               centerlines: torch.Tensor = None) -> np.ndarray:

    negpair = torch.tensor([-1,-1]).to("cuda")

    probabilities = probabilities.to("cuda")
    centerlines   = centerlines.to("cuda")

    targets = []
    for i in [1,2]:
        target = torch.stack([torch.argwhere(centerlines[b,0,h] == i)[0] if len(torch.argwhere(centerlines[b,0,h]==i))==1 else torch.tensor([-1,-1]).to("cuda") for b in range(centerlines.shape[0]) for h in range(centerlines.shape[2])]).to("cuda")
        target = target.view(-1, probabilities.shape[2], 2).type(torch.cuda.FloatTensor)
        targets.append(target)

    y = torch.arange(probabilities.shape[-2])
    x = torch.arange(probabilities.shape[-1])
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

    grid_y, grid_x = grid_y.to("cuda"), grid_x.to("cuda")

    l2dist = nn.PairwiseDistance(p=2, eps=eps)

    probs  = [probabilities, probabilities]
    loss = torch.tensor(0.0).to("cuda")

    for prob, target in zip(probs, targets):
        y_cnt = (grid_y*prob).sum(axis=(1,3,4)) / (prob.sum(axis=(1,3,4)) + eps)
        x_cnt = (grid_x*prob).sum(axis=(1,3,4)) / (prob.sum(axis=(1,3,4)) + eps)
        cnt  = torch.cat((y_cnt[:,:,None], x_cnt[:,:,None]),axis=2)

        l2 = l2dist(cnt, target) * (target != negpair)[:,:,0]
        loss += l2.mean()

    return loss.cpu().numpy()



def radius_constraint_metric(probabilities: torch.Tensor,
               eps: float = 1e-8,
               centerlines: torch.Tensor = None,
               radiuses: np.array = None) -> np.ndarray:

    negpair = torch.tensor([-1,-1]).to("cuda")
    relu = nn.ReLU()
    l2dist = nn.PairwiseDistance(p=2, eps=eps)
    alpha = 6

    y = torch.arange(probabilities.shape[-2])
    x = torch.arange(probabilities.shape[-1])
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    grid = torch.cat((grid_y[:,:,None], grid_x[:,:,None]), axis=2)
    grid = repeat(grid,"w h yx -> w h b d yx", b=probabilities.shape[0], d= probabilities.shape[2]).to("cuda")

    probabilities = probabilities.to("cuda")
    centerlines   = centerlines.to("cuda")

    targets = []
    for i in [1,2]:
        target = torch.stack([torch.argwhere(centerlines[b,0,h] == i)[0] if len(torch.argwhere(centerlines[b,0,h]==i))==1 else torch.tensor([-1,-1]).to("cuda") for b in range(centerlines.shape[0]) for h in range(centerlines.shape[2])]).to("cuda")
        target = target.view(-1, probabilities.shape[2], 2).type(torch.cuda.FloatTensor)
        targets.append(target)

    probs  = [probabilities, probabilities]
    lds = [1, 25]
    loss = torch.tensor(0.0).to("cuda")

    for prob, target, radius, ld in zip(probs, targets, radiuses, lds):
        # Masking where there's no center line
        mask = (target != negpair)
        mask = repeat(mask[:,:,0], "b z -> b c z y x", c=1, y = prob.shape[3], x =prob.shape[4]) 

        # calculate L2 distance
        d2 = l2dist(grid, target)

        # Match the shape for matmul
        d2 = rearrange(d2, "w h (b c) d -> b c d w h", c=1)
        # Calculate Exponential distance for out boundary
        dout = relu((torch.exp((d2/radius[1] - 1)/alpha) - 1)*mask)
        rc_out = (dout*prob).sum(axis=(1,3,4)) / (prob.sum(axis=(1,3,4)) + eps)

        # Calculate Exponential distance for in boundary
        ''' lumen's radius = outer wall's radius - 3'''
        lu_din  = relu((torch.exp(1-d2/radius[0])-1)*mask)
        ow_din  = relu((torch.exp(1-d2/radius[1])-1)*mask)
        lu_rc_in = (lu_din*(1-prob[:,:1])).sum(axis=(1,3,4)) / (radius[0] * radius[0] * torch.pi)
        ow_rc_in = (ow_din*(1-prob[:,1:])).sum(axis=(1,3,4)) / (radius[1] * radius[1] * torch.pi)
        rc_in = lu_rc_in + ow_rc_in

        loss += ld*(rc_out.mean() + rc_in.mean())

    return loss.cpu().numpy()


class Meter:
    '''factory for storing and updating dice, bce, cent and cnst scores.'''
    def __init__(self, 
                 treshold: float = 0.5):
        self.threshold: float = treshold
        self.dice_scores: list = []
        self.bce_scores: list = []
        self.cent_scores: list = []
        self.cnst_scores: list = []
    
    def update(self, 
               logits: torch.Tensor, 
               targets: torch.Tensor, 
               maskmats: torch.Tensor,
               centerlines: torch.Tensor,
               radiuses: np.array):
        """
        Takes: logits from output model and targets,
        calculates dice, bce, cent and cnst scores, and stores them in lists.
        """

        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, maskmats=maskmats)
        bce  = binary_cross_entropy_metric(logits, targets, maskmats=maskmats)
        cent = center_distance_metric(probs, centerlines=centerlines)
        cnst = radius_constraint_metric(probs, centerlines=centerlines, radiuses=radiuses)
        
        self.dice_scores.append(dice)
        self.bce_scores.append(bce)
        self.cent_scores.append(cent)
        self.cnst_scores.append(cnst)
    
    def get_metrics(self) -> np.ndarray:
        """
        Returns: the average of the accumulated dice, bce, cent and cnst scores.
        """
        dice = np.mean(self.dice_scores)
        bce  = np.mean(self.bce_scores)
        cent  = np.mean(self.cent_scores)
        cnst  = np.mean(self.cnst_scores)

        return dice, bce, cent, cnst


class MaskedDiceLoss(nn.Module):
    """Calculate DICE score"""
    def __init__(self, eps: float = 1e-8):
        """
        Params:
            eps: Epslion, default = 1e-8
        """
        super(MaskedDiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                maskmats: torch.Tensor) -> torch.Tensor:
        """
        Params:
            logits: model outputs of shape (b, c, z, y, x)
            probability: model outputs after sigmoid function. shape is (b, c, z, y, x)
            targets: ground truth values of shape (b, c, z, y, x)
            maskmats: Mask matrix for slice with annotation. shape is (b, c, z, y, x)

        Returns: 1 - DICE score. It must be scalar variable
        """

        assert(logits.shape == targets.shape)

        num = targets.size(0)
        probability = torch.sigmoid(logits)

        probability = probability.view(num, -1)
        targets     = targets.view(num, -1)
        maskmats   = maskmats.view(num,-1)
        
        targets     = torch.mul(targets, maskmats)
        probability = torch.mul(probability, maskmats)
        
        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / (union + self.eps)
        
        return 1.0 - dice_score


class MaskedDiceBCELoss(nn.Module):
    """
    Calculate BCE loss + DICE loss for data batch.
    """
    def __init__(self):
        """
        Params:
            dice: Dice loss with masked matrix
            eps: Epslion, default = 1e-8
        """
        super(MaskedDiceBCELoss, self).__init__()

        self.dice = MaskedDiceLoss()
        self.eps = 1e-8
        
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor,
                maskmats: torch.Tensor) -> torch.Tensor:
        """
        Params:
            logits: model outputs of shape (b, c, z, y, x)
            targets: ground truth values of shape (b, c, z, y, x)
            maskmats: Mask matrix for slice with annotation. shape is (b, c, z, y, x)

        Returns: BCE loss + DICE loss. It must be scalar variable
        """
        assert(logits.shape == targets.shape)

        bce_loss = nn.BCEWithLogitsLoss(pos_weight=maskmats)(logits, targets)
        dice_loss = self.dice(logits, targets, maskmats)
        
        return bce_loss + dice_loss



class CenterDistanceLoss(nn.Module):
    """
    Calculate center distance loss for data batch.
    """
    def __init__(self, shape, device):
        """
        Params:
            grid_y, grid_x: meshgrid
            device: Variables for parameters optimized for cuda
            eps: Epsilon, default = 1e-8
            l2dist: Pair wise Distance(L2 Distance)
            negpair: Mask for region without center line
        """
        super(CenterDistanceLoss, self).__init__()

        y = torch.arange(shape[0])
        x = torch.arange(shape[1])
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

        self.grid_y, self.grid_x = grid_y.to(device), grid_x.to(device)
        self.grid = torch.cat((self.grid_y[:,:,None], self.grid_x[:,:,None]), axis=2).to(device)
        self.device = device
        self.eps = 1e-6
        self.l2dist = nn.PairwiseDistance(p=2, eps=self.eps)
        self.negpair = torch.tensor([-1,-1]).to(device)
        self.relu  = nn.ReLU()

    def forward(self,
                probs: torch.Tensor,
                targets: list[torch.Tensor, torch.Tensor],
                radiuses: np.array,
                ) -> torch.Tensor:
        """
        Params:
            probs: model outputs after sigmoid function [probs_i, probs_e] , probs = sigmoid(logits)
                probs_i: probability of ICA
                probs_e: probability of ECA
            targets: [target_i, target_e]
                target_i: target center points of ICA
                target_e: target center points of ECA

        Returns: L2 Distance between predict and target
        """

        loss = torch.tensor(0.0).to(self.device)

        art_masks = []
        grid = repeat(self.grid, "w h yx -> w h b d yx", b=probs.shape[0], d= probs.shape[2])
        for target, radius in zip(targets, radiuses):
            # calculate L2 distance
            d2 = self.l2dist(grid, target)
            # Match the shape for matmul
            d2 = rearrange(d2, "w h (b c) d -> b c d w h", c=1)
            # Calculate Exponential distance for out boundary
            dout = self.relu((torch.exp((d2/radius[1] - 1)) - 1))
            dout[dout > 0] = 1
            art_masks.append(dout)

        art_masks = [art_masks[1], art_masks[0]]
        
        for target, art_mask in zip(targets, art_masks):
            mask = (target != self.negpair)[:,:,0]

            y_cnt = ((art_mask*self.grid_y)*probs).sum(axis=(1,3,4)) / ((probs*art_mask).sum(axis=(1,3,4)) + self.eps)
            x_cnt = ((art_mask*self.grid_x)*probs).sum(axis=(1,3,4)) / ((probs*art_mask).sum(axis=(1,3,4)) + self.eps)
            cnt  = torch.cat((y_cnt[:,:,None], x_cnt[:,:,None]),axis=2)

            l2dist = self.l2dist(cnt, target) * mask
            loss = loss + l2dist.mean()

        return loss


class RadiusConstraintLoss(nn.Module):
    """
    Calculate Proximity loss.
    Loss that constraint the predicted voxel at the center point to be within the radius
    """
    def __init__(self, shape, device):
        """
        Params:
            grid, grid_y, grid_x: meshgrid
            device: Variables for parameters optimized for cuda
            alpha: scaling factor for the distance out boundary
            eps: Epsilon, default = 1e-8
            l2dist: Pair wise Distance(L2 Distance)
            negpair: Mask for region without center line
            relu: nn.ReLU()
        """
        super(RadiusConstraintLoss, self).__init__()

        y = torch.arange(shape[0])
        x = torch.arange(shape[1])

        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.grid = torch.cat((grid_y[:,:,None], grid_x[:,:,None]), axis=2).to(device)
        self.device = device
        self.alpha = 5
        self.eps   = 1e-6
        self.negpair = torch.tensor([-1,-1]).to(device)
        self.l2dist = nn.PairwiseDistance(p=2, eps=self.eps)
        self.relu  = nn.ReLU()

    def forward(self,
                probs: torch.Tensor,
                targets: list[torch.Tensor, torch.Tensor],
                radiuses: np.array,
                ) -> torch.Tensor:
        """
        Params:
            probs: model outputs after sigmoid function [probs_i, probs_e] , probs = sigmoid(logits)
                probs_i: probability of ICA
                probs_e: probability of ECA
            targets: [target_i, target_e]
                target_i: target center points of ICA
                target_e: target center points of ECA
            radius: [radius_i, radius_e]
                radius_i: Constrainted radius of ICA
                radius_e: Constrainted radius of ECA
            mask: no calculate where there is no center line. Created a mask with negpair.
            d2: Distance to center point each pixel (L2 Distance)
            dout: Distance map giving loss to pixels outside radius
            rc_out: Radius Constraint loss for pixels outside radius
                    Suppresses false positive predictions outside radius
            din: Distance map giving loss to pixels inside radius
                lu_din: Distance map for lumen's radius
                ow_din: Distance map for outer wall's radius
            rc_in: Radius Constraint loss for pixels inside radius
                   Encourage prediction inside radius
                lu_rc_in: for pixels inside lumen's radius
                ow_rc_in: for pixels inside outer wall's radius
            loss: Radius Constraint loss_in + Radius Constraint loss_out
            
        Returns: loss
        """

        loss = torch.tensor(0.0).to(self.device)
        
        art_masks = []
        grid = repeat(self.grid, "w h yx -> w h b d yx", b=probs.shape[0], d= probs.shape[2])
        for target, radius in zip(targets, radiuses):
            # calculate L2 distance
            d2 = self.l2dist(grid, target)

            # Match the shape for matmul
            d2 = rearrange(d2, "w h (b c) d -> b c d w h", c=1)
            # Calculate Exponential distance for out boundary
            dout = self.relu((torch.exp((d2/radius[1] - 1)) - 1))
            dout[dout > 0] = 1
            art_masks.append(dout)

        art_masks = [art_masks[1], art_masks[0]]

        for target, radius, art_mask in zip(targets, radiuses, art_masks):
            # Masking where there's no center line
            mask = (target != self.negpair)
            mask = repeat(mask[:,:,0], "b z -> b c z y x", c=1, y = probs.shape[3], x =probs.shape[4]) 

            # calculate L2 distance
            d2 = self.l2dist(grid, target)

            # Match the shape for matmul
            d2 = rearrange(d2, "w h (b c) d -> b c d w h", c=1)

            # Calculate Exponential distance for out boundary
            lu_dout = self.relu(1 - torch.exp(1 -d2/radius[0]))/self.alpha * mask
            ow_dout = self.relu(1 - torch.exp(1 -d2/radius[1]))/self.alpha * mask
            lu_dout = lu_dout * art_mask
            ow_dout = ow_dout * art_mask
            lu_rc_out = (lu_dout*probs[:,:1]).sum(axis=(1,3,4)) / (probs[:,:1].sum(axis=(1,3,4)) + self.eps)
            ow_rc_out = (ow_dout*probs[:,1:]).sum(axis=(1,3,4)) / (probs[:,1:].sum(axis=(1,3,4)) + self.eps)
            rc_out = lu_rc_out + ow_rc_out

            # Calculate Exponential distance for in boundary
            ''' lumen's radius = outer wall's radius - 3'''
            lu_din  = self.relu(torch.exp(1 - d2/radius[0]) - 1) * mask
            ow_din  = self.relu(torch.exp(1 - d2/radius[1]) - 1) * mask
            lu_rc_in = (lu_din*(1-probs[:,:1])).sum(axis=(1,3,4)) / (radius[0] * radius[0] * torch.pi)
            ow_rc_in = (ow_din*(1-probs[:,1:])).sum(axis=(1,3,4)) / (radius[1] * radius[1] * torch.pi)
            rc_in = lu_rc_in + ow_rc_in

            loss = loss + rc_out.mean() + rc_in.mean()

        return loss


class ConsistencyLoss(nn.Module):
    """Calculate Consistency loss."""
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        """
        Params:
            T: Temperature parameter
        """
        self.T = 10

    def forward(self, logits: torch.Tensor) -> torch.Tensor:        
        """
        Params:
            P_orig : Probability map of original slices 
            P_shift: Probability map of shifted slices 
            loss: Binary cross entropy loss between two distilled probability maps

        Returns: loss
        """

        # Gradinet does not flow in the logit of the truth role.
        P_orig  = F.softmax(logits[:,:,1:-1]/self.T).detach()
        P_shift = F.softmax(logits[:,:,0:-2]/self.T)

        loss = F.binary_cross_entropy(P_orig, P_shift)

        return loss
        


class TotalLoss(nn.Module):
    """Compute objective loss: CenterDistance Loss + RadiusConstraint Loss + Consistency Loss"""
    def __init__(self, shape, device, mode):
        super(TotalLoss, self).__init__()
        """
        Params:
            cent: Center Distance Loss
            cnst: Radius Constraint Loss
            csis: Consistency Loss
        """
        self.mode = mode
        self.device = device
        self.cent = CenterDistanceLoss(shape, device)
        self.cnst = RadiusConstraintLoss(shape, device)
        self.csis = ConsistencyLoss()

    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor,
                maskmats: torch.Tensor,
                centerlines: torch.Tensor,
                radiuses: np.array,
                epoch: int) -> torch.Tensor:
        """
        Params:
            ld: lambda, Weight of each loss
            loss: CenterDistance Loss + RadiusConstraint Loss + Consistency Loss
        Returns: loss
        """
        ld1, ld2, ld3 = 0.02, 1, 0.001

        probs = torch.sigmoid(logits)
        targets = []

        for i in [1,2]:
            target = torch.stack([
                torch.argwhere(centerlines[b,0,h] == i)[0] 
                if len(torch.argwhere(centerlines[b,0,h]==i))==1 
                else torch.tensor([-1,-1]).to(self.device)
                for b in range(centerlines.shape[0]) 
                for h in range(centerlines.shape[2])
            ]).to(self.device)

            target = target.view(-1, logits.shape[2], 2).type(torch.cuda.FloatTensor)
            targets.append(target)

        loss_cent = self.cent(probs, targets=targets, radiuses=radiuses)
        loss_cnst = self.cnst(probs, targets=targets, radiuses=radiuses)
        loss_csis = self.csis(logits)
        
        if self.mode == "centd":
            loss = loss_cent
        elif self.mode == "cnst":
            loss = ld2*loss_cnst
        elif self.mode == "centd_cnst":
            loss = ld1*loss_cent + ld2*loss_cnst
        elif self.mode == "total":
            loss = ld1*loss_cent + ld2*loss_cnst + ld3*loss_csis  

        return loss, loss

    