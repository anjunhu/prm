try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import copy
import itertools
import logging
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from collections import OrderedDict, defaultdict, abc
from typing import Any, Dict, List, Set, Union
import time 
import datetime
import random
from contextlib import ExitStack, contextmanager

import detectron2.utils.comm as comm
import torch
import torchvision

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel")
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesSemSegEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
    print_csv_format,
    DatasetEvaluator
)
from detectron2.structures import ImageList
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from tabulate import tabulate

from san import (
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_san_config,
)
from san.data import build_detection_test_loader, build_detection_train_loader
from san.utils import WandbWriter, setup_wandb
import torch.nn.functional as F
from torchvision import transforms

import clip
import open_clip

def grab_clip_features_vit(inputs, clip_model):
    grid_size = round(( clip_model.visual.positional_embedding.shape[0] - 1) ** 0.5)
    x = inputs["clip_input"]
    all_clip_features = []
    x = clip_model.visual.conv1(x)  # shape = [*, width, grid, grid]
    b, _, gh, gw = x.size()
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    if x.shape[1] != clip_model.visual.positional_embedding.shape[0]:
        positional_embedding = clip_model.visual.positional_embedding
        # print(positional_embedding.shape, self.grid_size) #[197, 768], 14
        if not (clip_model.visual.positional_embedding.shape[0] == x.shape[1]):
            cls_pos = positional_embedding[0:1, :]
            per_pos_embedding = (
                    F.interpolate(
                        positional_embedding[1:, :]
                        .permute(1, 0)
                        .view(1, -1, grid_size, grid_size),
                        size=(gh, gw),
                        mode="bicubic",
                    )
                    .reshape(-1, gh * gw)
                    .permute(1, 0)
                )
            # print(per_pos_embedding.shape) #[gh*gw, 768]

            positional_embedding = torch.cat([cls_pos, per_pos_embedding])
        x = x + positional_embedding.to(x.dtype)
    else:
        x = x + clip_model.visual.positional_embedding.to(x.dtype)

    x = clip_model.visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND
    all_clip_features.append(x)
    for i, resblock in enumerate(clip_model.visual.transformer.resblocks):
        x = resblock(x)
        all_clip_features.append(x)
    return all_clip_features


def grab_clip_features_rn(inputs, clip_model):
    x = inputs["clip_input"]
    all_clip_features = []
    x = clip_model.visual.relu1(clip_model.visual.bn1(clip_model.visual.conv1(x)))
    x = clip_model.visual.relu2(clip_model.visual.bn2(clip_model.visual.conv2(x)))
    x = clip_model.visual.relu3(clip_model.visual.bn3(clip_model.visual.conv3(x)))
    x = clip_model.visual.avgpool(x)

    all_clip_features.append(x) # B, 96, H, W
    x = clip_model.visual.layer1(x) # B, 384, H, W
    all_clip_features.append(x)
    x = clip_model.visual.layer2(x) # B, 768,, H, W
    all_clip_features.append(x)
    x = clip_model.visual.layer3(x) # B, 1536, H, W
    all_clip_features.append(x)
    x = clip_model.visual.layer4(x) # B, 3072, H, W
    all_clip_features.append(x)

    gh, gw = x.shape[-2:]
    original_spacial_dim = int((clip_model.visual.attnpool.positional_embedding.shape[0]-1) ** 0.5)
    x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
    if x.shape[0] != clip_model.visual.attnpool.positional_embedding.shape[0]:
        cls_pos = clip_model.visual.attnpool.positional_embedding[0:1, :]
        per_pos_embedding = (
                F.interpolate(
                    clip_model.visual.attnpool.positional_embedding[1:, :]
                    .permute(1, 0)
                    .view(1, -1, original_spacial_dim, original_spacial_dim),
                    size=(gh, gw),
                    mode="bicubic",
                )
                .reshape(-1, gh * gw)
                .permute(1, 0)
            )
        # print(cls_pos.shape, per_pos_embedding.shape)
        positional_embedding = torch.cat([cls_pos, per_pos_embedding])[:, None, :]
        x += positional_embedding
    else:
        x = x + clip_model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
    x, _ = F.multi_head_attention_forward(
        query=x[:1], key=x, value=x,
        embed_dim_to_check=x.shape[-1],
        num_heads=clip_model.visual.attnpool.num_heads,
        q_proj_weight=clip_model.visual.attnpool.q_proj.weight,
        k_proj_weight=clip_model.visual.attnpool.k_proj.weight,
        v_proj_weight=clip_model.visual.attnpool.v_proj.weight,
        in_proj_weight=None,
        in_proj_bias=torch.cat([clip_model.visual.attnpool.q_proj.bias,
                                clip_model.visual.attnpool.k_proj.bias,
                                clip_model.visual.attnpool.v_proj.bias]),
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0,
        out_proj_weight=clip_model.visual.attnpool.c_proj.weight,
        out_proj_bias=clip_model.visual.attnpool.c_proj.bias,
        use_separate_proj_weight=True,
        training=clip_model.visual.attnpool.training,
        need_weights=False
    )
    return all_clip_features

def grab_clip_features_convnext(inputs, clip_model):
    x = inputs["clip_input"]
    all_clip_features = []
    x = clip_model.visual.trunk.stem(x)
    all_clip_features.append(x)
    # print(x.shape)
    for s, stage in enumerate(clip_model.visual.trunk.stages):
        x = stage.downsample(x)
        for b, block in enumerate(stage.blocks):
            x = block(x)
            all_clip_features.append(x)
            # print(f'Stage {s} Block {b}', x.shape)
    x = clip_model.visual.trunk.norm_pre(x)
    return all_clip_features

CLIP_PIXEL_MEAN = torch.Tensor([122.7709383, 116.7460125, 104.09373615]).view(-1, 1, 1) 
CLIP_PIXEL_STD = torch.Tensor([68.5005327, 66.6321579, 70.3231630]).view(-1, 1, 1) 

def clip_preprocess(batched_inputs, device):
    images = [x["image"].to(device) for x in batched_inputs]
    images = [(x - CLIP_PIXEL_MEAN.to(device)) / CLIP_PIXEL_STD.to(device) for x in images]
    images = ImageList.from_tensors(images, -1)
    return {"clip_input": images.tensor}
    
def attack_dataset(
    cfg,
    args,
    model,
    data_loader,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    callbacks=None,
):
    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))
    random.seed(cfg.SEED)

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    os.makedirs(f'{cfg.OUTPUT_DIR}/images/val_adv/', exist_ok=True)
    os.makedirs(f'{cfg.OUTPUT_DIR}/images/val_adv_visuals/', exist_ok=True)

    with ExitStack() as stack:
        model.eval()
        model.requires_grad_(False)

        start_data_time = time.perf_counter()
        dict.get(callbacks or {}, "on_start", lambda: None)()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            dict.get(callbacks or {}, "before_inference", lambda: None)()

            # Get clean outputs for reference, visualisation and relativistic losses
            ori_sizes = [x["ori_size"] for x in inputs]
            fns = [x['file_name'].split('/')[-1] for x in inputs] 
            if fns[0].split('.')[0] + '.pt' in os.listdir(f'{cfg.OUTPUT_DIR}/images/val_adv/'):
                print(fns[0].split('.')[0] + '.pt already created, skipping')
                continue
            clean_images = [x["image"].float().detach().clone().to(model.device) for x in inputs]
            perturbation = [torch.zeros_like(x["image"]).float().to(model.device) for x in inputs]
            unscaled_gts = [copy.deepcopy(x["instances"]) for x in inputs]
            for p in perturbation:
                p.requires_grad_(True) #torch.Size([3, H, W]) 
            adv_inputs = copy.deepcopy(inputs)

            attack_optimizer = torch.optim.AdamW(perturbation, lr=5e-1)

            if 'convnext' not in args.model:
                clip_model, _ = clip.load(args.model, device=model.device)
                grab_clip_features = grab_clip_features_rn if 'rn' in args.model else grab_clip_features_vit
            elif 'convnext_large'in args.model:
                clip_model, _, _ = open_clip.create_model_and_transforms('convnext_large_d_320',
                                            pretrained="laion2b_s29b_b131k_ft_soup", device=model.device)
                grab_clip_features = grab_clip_features_convnext
            elif 'convnext_base'in args.model:
                clip_model, _, _ = open_clip.create_model_and_transforms('convnext_base_w_320',
                                            pretrained="laion_aesthetic_s13b_b82k", device=model.device)
                grab_clip_features = grab_clip_features_convnext
            clip_model = clip_model.float()
            clip_model.requires_grad_(False)

            for iter in range(250):
                for s, _ in enumerate(perturbation): 
                    ### Inject perturabation
                    adv_inputs[s]["image"] = (clean_images[s] + perturbation[s].clamp(-8., 8.)).clamp(0., 255.)

                    if args.dynamic_scale:
                        ratio = torch.rand((1,)).to(adv_inputs[s]["image"].device)*0.5 + 0.5
                        target_scale = (int(ratio*ori_sizes[s][0]), int(ratio*ori_sizes[s][1]))
                        adv_inputs[s]["image"] = F.interpolate(adv_inputs[s]["image"].unsqueeze(0), target_scale, mode='bicubic').squeeze(0)
                        adv_inputs[s]["instances"].gt_masks = F.interpolate(unscaled_gts[s].gt_masks.float().unsqueeze(0), target_scale, mode='bicubic').squeeze(0).bool()
                        inputs[s]["image"] = F.interpolate(clean_images[s].unsqueeze(0), target_scale, mode='bicubic').squeeze(0)
                        inputs[s]["instances"].gt_masks = F.interpolate(unscaled_gts[s].gt_masks.float().unsqueeze(0), target_scale, mode='bicubic').squeeze(0).bool()

                outputs = clip_preprocess(adv_inputs, model.device)
                clean_outputs = clip_preprocess(inputs, model.device)

                adv_clip_features = grab_clip_features(outputs, clip_model)
                clean_clip_features = grab_clip_features(clean_outputs, clip_model)

                losses = 0.0
                for layer, (item, _) in enumerate(zip(adv_clip_features, clean_clip_features)):
                    losses -= torch.std(item)

                attack_optimizer.zero_grad()
                losses.backward()
                attack_optimizer.step()

            dict.get(callbacks or {}, "after_inference", lambda: None)()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            for s, _ in enumerate(perturbation): 
                adv_inputs[s]["image"] = (clean_images[s] + perturbation[s].clamp(-8., 8.)).clamp(0., 255.)
            processed_results = model(adv_inputs)
            evaluator.process(inputs, processed_results)
            total_eval_time += time.perf_counter() - start_eval_time
                            
            for i, x in enumerate(adv_inputs):
                images = x["image"].float().detach().unsqueeze(0).cpu()
                # images = images[:, :, :ori_sizes[i][0], :ori_sizes[i][1]]
                fn = x['file_name'].split('/')[-1]
                torch.save(images, f'{cfg.OUTPUT_DIR}/images/val_adv/{fn[:-4]}.pt')
                print(f'{cfg.OUTPUT_DIR}/images/val_adv/{fn[:-4]}.pt')
                torchvision.utils.save_image(images/255., f'{cfg.OUTPUT_DIR}/images/val_adv_visuals/{fn}')

        dict.get(callbacks or {}, "on_end", lambda: None)()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


class Trainer(DefaultTrainer):
    def build_writers(self):
        writers = super().build_writers()
        # use wandb writer instead.
        writers[-1] = WandbWriter()
        return writers

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # resue maskformer dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, False)
        else:
            mapper = None
        # Add dataset meta info.
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        # use poly scheduler
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed_group = cfg.SOLVER.WEIGHT_DECAY_EMBED_GROUP
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                hyperparams["param_name"] = ".".join([module_name, module_param_name])
                if "side_adapter_network" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                # scale clip lr
                if "clip" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.CLIP_MULTIPLIER
                if any([x in module_param_name for x in weight_decay_embed_group]):
                    hyperparams["weight_decay"] = weight_decay_embed
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        # display the lr and wd of each param group in a table
        optim_info = defaultdict(list)
        total_params_size = 0
        for group in optimizer.param_groups:
            optim_info["Param Name"].append(group["param_name"])
            optim_info["Param Shape"].append(
                "X".join([str(x) for x in list(group["params"][0].shape)])
            )
            total_params_size += group["params"][0].numel()
            optim_info["Lr"].append(group["lr"])
            optim_info["Wd"].append(group["weight_decay"])
        # Counting the number of parameters
        optim_info["Param Name"].append("Total")
        optim_info["Param Shape"].append("{:.2f}M".format(total_params_size / 1e6))
        optim_info["Lr"].append("-")
        optim_info["Wd"].append("-")
        table = tabulate(
            list(zip(*optim_info.values())),
            headers=optim_info.keys(),
            tablefmt="grid",
            floatfmt=".2e",
            stralign="center",
            numalign="center",
        )
        logger = logging.getLogger("san")
        logger.info("Optimizer Info:\n{}\n".format(table))
        return optimizer
    
    @classmethod
    def attack(cls, cfg, args, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = attack_dataset(cfg, args, model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    if not args.eval_only:
        setup_wandb(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="prm_attack")
    return cfg


def main(args):
    cfg = setup(args)

    if args.adv_attack:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.attack(cfg, args, model)
        return res
    elif args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--adv-attack",
        action="store_true",
    )
    parser.add_argument(
        "--dynamic-scale",
        action="store_true",
    )
    parser.add_argument(
        "--model",
        default="ViT-B/16",
        choices=["ViT-B/16","ViT-L/14","RN50","convnext_large","convnext_base"]
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
