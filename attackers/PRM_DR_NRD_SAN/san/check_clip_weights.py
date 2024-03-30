import torch
import clip

checkpoint = torch.load('./pretrained_models/san_vit_b_16.pth')
clip_model, preprocess = clip.load("ViT-B/16")

clip_visual_state_dict = {}
for key in checkpoint['model'].keys():
    if 'clip_visual_extractor' in key:
        clip_visual_state_dict[key[22:]] = checkpoint['model'][key]

clip_visual_original_state_dict = {}
for key in clip_model.state_dict().keys():
    if 'visual.transformer' in key:
        clip_visual_original_state_dict[key[19:]] = clip_model.state_dict()[key]
    elif 'visual.' in key:
        clip_visual_original_state_dict[key[7:]] = clip_model.state_dict()[key]

overlap = set(clip_visual_state_dict).intersection(set(clip_visual_original_state_dict))
print(len(overlap), len(clip_visual_state_dict ))

for key in overlap:
    print(key, torch.equal(clip_visual_state_dict[key], clip_visual_original_state_dict[key]))