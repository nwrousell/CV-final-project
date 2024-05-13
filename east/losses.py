import torch
from torch.nn import CrossEntropyLoss

from .icdar import restore_rectangle_rbox


def compute_dice_loss(pred_score, gt_score):
    intersection_area = torch.sum(gt_score * pred_score)
    sum_of_areas = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
    return 1 - ((2 * intersection_area) / sum_of_areas)

class EastLoss:
    def __init__(self, geometry_loss_weight, angle_loss_weight):
        # weight = torch.tensor([fraction_negative, 1-fraction_negative])
        # self.balanced_cross_entropy_loss = CrossEntropyLoss(weight=weight)

        self.geometry_loss_weight = geometry_loss_weight
        self.angle_loss_weight = angle_loss_weight

    def __call__(self, score_label, pred_score, geo_label, pred_geometry, training_mask):
        # score map loss (eq5)
        # score_loss = self.balanced_cross_entropy_loss(pred_score, score_label)

        # score map (dice loss instead)
        inp = pred_score*(1-training_mask)
        score_loss = compute_dice_loss(score_label , pred_score*training_mask)

        # geometry map loss
        d1_gt, d2_gt, d3_gt, d4_gt, rotation_angle_label = torch.split(geo_label, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred, rotation_angle_pred = torch.split(pred_geometry, 1, 1)

        area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
        area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)

        # eq8 
        w_intersect = torch.minimum(d2_gt, d2_pred) + torch.minimum(d4_gt, d4_pred)
        h_intersect = torch.minimum(d1_gt, d1_pred) + torch.minimum(d3_gt, d3_pred)
        area_intersect = w_intersect * h_intersect

        # eq 9
        area_union = (area_pred + area_gt) - area_intersect

        # eq 7
        IoU_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        # eq10
        rotation_angle_loss_map = 1 - torch.cos(rotation_angle_pred - rotation_angle_label)

        # only count geometry loss over bounding boxes
        IoU_loss = torch.sum(IoU_loss_map * score_label) / torch.sum(score_label)
        rotation_angle_loss = torch.sum(rotation_angle_loss_map * score_label) / torch.sum(score_label)

        geometry_loss = IoU_loss + self.angle_loss_weight * rotation_angle_loss

        total_loss = score_loss + self.geometry_loss_weight * geometry_loss
        return total_loss