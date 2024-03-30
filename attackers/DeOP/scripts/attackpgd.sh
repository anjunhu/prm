# pip uninstall clip -y
# pip install third_party/CLIP/

CUDA_VISIBLE_DEVICES="1,5"
proposalmodel="pretrained_models/deop_model_final.pth"
OutPutDir="attack_coco_pgd_deop"
promptlearn="learnable"
configfile="configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k_proposalmask_featupsample_img512.yaml"
maskformermodel="ZeroShotClipfeatUpsample_addnorm_vit16Query2D_maskvit"

python3 attack_PGD.py --config-file ${configfile} --num-gpus 2 --eval-only --adv-attack --dist-url tcp://0.0.0.0:12445 \
 SEED 2023 SOLVER.TEST_IMS_PER_BATCH 1 OUTPUT_DIR ${OutPutDir} MODEL.CLIP_ADAPTER.PROMPT_LEARNER ${promptlearn} \
 MODEL.WEIGHTS ${proposalmodel} MODEL.META_ARCHITECTURE ${maskformermodel} \
 MODEL.NUM_DECODER_LAYER 1 ORACLE False \
 MODEL.MASK_FORMER.DECODER_DICE_WEIGHT 0.8 MODEL.MASK_FORMER.DECODER_CE_WEIGHT 2.0 \
 MODEL.CLIP_ADAPTER.LEARN_TOKEN False \
 MODEL.CLIP_ADAPTER.LEARN_POSITION True \
 MODEL.CLIP_ADAPTER.POSITION_LAYERS '[1,2,3,4,5]' \
 MODEL.CLIP_ADAPTER.LAYERMASKVIT '[11,]' \
 DATASETS.TEST "('coco_2017_train_stuff_all_sem_seg',)" \
 MODEL.SEM_SEG_HEAD.NUM_CLASSES 171  MODEL.MASK_FORMER.LOSS_NUM_CLASS 171
