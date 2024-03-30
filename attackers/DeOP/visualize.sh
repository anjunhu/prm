basename=${1:-""}
outdir="visuals-prm"
mkdir -p ${outdir}

model_path="pretrained_models/deop_model_final.pth"
configfile="configs/attack-eval-visualize.yaml"

for file in $(ls $basename)
do
image=`basename -- "$file" .pt
#image=${image::-3}`
echo $image
python visualize.py \
 --config-file $configfile \
 --model-path $model_path \
 --img-path ${basename}/${image}.pt \
 --output-file ./${outdir}/${image}_adv

python visualize.py \
 --config-file $configfile \
 --model-path $model_path \
 --img-path ../..//datasets/VOCdevkit/VOC2010/JPEGImages/${image}.jpg \
 --output-file ./${outdir}/${image}_clean
done

