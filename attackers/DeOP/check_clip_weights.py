import torch
import clip

# checkpoint = torch.load('./deop/train_log/model_0002999.pth')
# clip_model, preprocess = clip.load("ViT-B/16")

# deop_model_state_dict = {}
# for key in checkpoint['model'].keys():
#     if 'clip_adapter.clip_model' in key and not 'learned_position' in key:
#         deop_model_state_dict[key] = checkpoint['model'][key]
#     else:
#         print(key)

# for k1, k2 in zip(deop_model_state_dict, clip_model.state_dict()):
#     assert k2 in k1
#     print(k2, deop_model_state_dict[k1].shape, torch.equal(clip_model.state_dict()[k2], deop_model_state_dict[k1]))

checkpoint = torch.load('./deop/lora_attn_mlp/model_0004999.pth', map_location="cpu")['model']
original = torch.load('pretrained_models/deop_model_final.pth', map_location="cpu")['model']
for key in checkpoint.keys():
    if key in original:
        if not torch.equal(checkpoint[key], original[key]):
            print('CHANGED', key)
    else:
        print('NEW', key)