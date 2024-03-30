This is an implementation of Patch Representation Misalignment (PRM), Neural Representation Distortion (NRD) and Dispersion Reduction (DR) attack based on [Detectron2](https://github.com/facebookresearch/detectron2/) and [SAN](https://github.com/MendelXu/SAN/tree/main) repositories.

`attack_*.py` follows an analogous structure as [Detectron2](https://github.com/facebookresearch/detectron2/) `train_net.py` and can be used as such. To reproduce, please run:
```
python attack_*.py --eval-only --adv-attack --dynamic-scale --dist-url tcp://0.0.0.0:12345 \
 --config-file $CONFIG_FILE --num-gpus $NUM_GPUS \
 SEED 2023 OUTPUT_DIR $OUTPUT_DIR \
 MODEL.WEIGHTS ./pretrained_models/san_vit_b_16.pth \
 DATASETS.TEST "('coco_2017_*_stuff_sem_seg',)"
```
Alternatively, please use `sh scripts/attack.sh` 
Saved aversarial datasets can be used for inference with other target models in `targets/`.