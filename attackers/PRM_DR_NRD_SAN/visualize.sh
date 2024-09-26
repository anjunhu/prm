vocab='aeroplane,wheel,truck,grass,field,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor'
image='2009_001279'

for file in $(ls $1)
do
image=`basename -- "$file" .pt
#image=${image::-3}`
echo $image
python predict.py \
 --config-file configs/san_clip_vit_res4_coco.yaml \
 --model-path ./pretrained_models/san_vit_b_16.pth \
 --img-path /scratch/local/ssd/anjun/datasets/VOCdevkit/VOC2010/JPEGImages_adv_deop_latest/$image.pt \
 --vocab $vocab \
 --output-file ./visual/output_adv_$image.png

python predict.py \
 --config-file configs/san_clip_vit_res4_coco.yaml \
 --model-path ./pretrained_models/san_vit_b_16.pth \
 --img-path /scratch/local/ssd/anjun/datasets/VOCdevkit/VOC2010/JPEGImages/$image.jpg \
 --vocab $vocab \
 --output-file ./visual/output_clean_$image.png
done
