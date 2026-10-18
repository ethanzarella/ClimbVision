from sam_wall_geometry import box_iou, anchor_agreement
from wall_tracker import transform_holds
import numpy as np

def test_bad_hand_mask_does_not_change_geometry():
    reviewed = [{'hold_id':'H13','x':100,'y':70,'w':24,'h':15}]
    H = np.array([[1.,0.,30.],[0.,1.,8.],[0.,0.,1.]])
    projected = transform_holds(reviewed, H, (260,360,3))
    # Occluding hand is much bigger, follows the body and not the wall.
    bad_sam = [{'hold_id':'H13','x':170,'y':30,'w':90,'h':110}]
    stats = anchor_agreement(projected, bad_sam)
    assert stats['sam_disagreed'] == 1
    assert (projected[0]['x'],projected[0]['y']) == (130,78)
    assert (projected[0]['w'],projected[0]['h']) == (24,15)

def test_sam_agreement_only_diagnostic():
    base = [{'hold_id':'H13','x':100,'y':70,'w':20,'h':20}]
    assert anchor_agreement(base,[dict(base[0])]) == {'sam_agreed':1,'sam_disagreed':0}
    assert box_iou(base[0],{'x':300,'y':300,'w':20,'h':20}) == 0
