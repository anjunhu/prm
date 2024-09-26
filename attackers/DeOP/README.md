This is an implementation of PGD attack based on [Detectron2](https://github.com/facebookresearch/detectron2/) and [DeOP](https://github.com/CongHan0808/DeOP) repositories.

`attack_PGD.py` follows an analogous structure as [Detectron2](https://github.com/facebookresearch/detectron2/) `train_net.py` and can be used as such. To reproduce, please run:
```
python attack_PGD.py --eval-only --adv-attack --dynamic-scale --dist-url tcp://0.0.0.0:12345 \
 --config-file $CONFIG_FILE --num-gpus $NUM_GPUS \
 SEED 2023 OUTPUT_DIR $OUTPUT_DIR \
 MODEL.WEIGHTS ./pretrained_models/san_vit_b_16.pth \
 DATASETS.TEST "('coco_2017_*_stuff_sem_seg',)"
```