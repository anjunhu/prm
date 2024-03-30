export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

od=output/attack_coco_prm_vitb16
attacker=attack_PRM.py #Choices: PRM, PGD, NRD, DR

python $attacker --eval-only --adv-attack --dynamic-scale  --dist-url tcp://0.0.0.0:12369 \
 --config-file configs/san_clip_vit_res4_coco.yaml --num-gpus 8 \
 SEED 2023 OUTPUT_DIR $od \
 MODEL.WEIGHTS ./pretrained_models/san_vit_b_16.pth \
 DATASETS.TEST "('coco_2017_test_stuff_all_sem_seg',)"
