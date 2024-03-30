# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
from functools import partial
import copy
import itertools
import logging
import os
from collections import OrderedDict, abc
from typing import Any, Dict, List, Set, Union
import time
import datetime
import random
from contextlib import ExitStack, contextmanager
import torchvision
import torch.nn.functional as F

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    create_ddp_model,
    AMPTrainer, SimpleTrainer, TrainerBase,
)
from detectron2.evaluation import (
    CityscapesSemSegEvaluator,
    DatasetEvaluators,
    verify_results,
    print_csv_format,
    DatasetEvaluator
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import CommonMetricPrinter, JSONWriter

# MaskFormer
from mask_former import SemanticSegmentorWithTTA, add_mask_former_config
from mask_former.data import (
    MaskFormerSemanticDatasetMapper,
    MaskFormerCaptionSemanticDatasetMapper,
    OracleDatasetMapper,
    MaskFormerBinarySemanticDatasetMapper,
    ProposalClasificationDatasetMapper,
)

from mask_former.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    dataset_sample_per_class,
)
from mask_former.evaluation import (
    GeneralizedSemSegEvaluator,
    GeneralizedPseudoSemSegEvaluator,
    ClassificationEvaluator,
    GeneralizedSemSegEvaluatorClassAgnosticSeenUnseen,
    GeneralizedSemSegEvaluatorClassAgnosticOutput,
)
from mask_former.utils.events import WandbWriter, setup_wandb
from mask_former.utils.post_process_utils import dense_crf_post_process

from mask_former.data.datasets.register_coco_stuff import COCO_CATEGORIES
from mask_former.data.datasets.register_pcontext import PCONTEXT_SEM_SEG_CATEGORIES
from mask_former.modeling.criterion import SetCriterion
from mask_former.modeling.matcher import HungarianMatcher

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def grab_clip_features(x, clip_model):
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
                        .view(1, -1, clip_model.visual.grid_size, clip_model.visual.grid_size),
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
    for i, resblock in enumerate(clip_model.visual.transformer.resblocks):
        x = resblock(x)
        all_clip_features.append(x)
    return all_clip_features


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
    seed_everything(cfg.SEED)

    # Loss parameters:
    deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
    no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
    dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
    mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
    kd_weight = cfg.MODEL.MASK_FORMER.CLIP_KD_WEIGHT
    loss_decoder_dice = cfg.MODEL.MASK_FORMER.DECODER_DICE_WEIGHT
    loss_ce = cfg.MODEL.MASK_FORMER.DECODER_CE_WEIGHT
    num_class = cfg.MODEL.MASK_FORMER.LOSS_NUM_CLASS

    # building criterion
    matcher = HungarianMatcher(
        # cost_class=1,
        cost_class=1,
        cost_mask=mask_weight,
        cost_dice=dice_weight,
    )

    weight_dict = {"loss_ce": loss_ce, "loss_mask": mask_weight,\
            "loss_dice": dice_weight, "loss_kd": kd_weight, "loss_decoder_dice": loss_decoder_dice}
    

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

    if args.filter_vqa_cap:
        import json
        f = open('/scratch/local/ssd/anjun/datasets/coco/okvqa/OpenEnded_mscoco_val2014_questions.json')
        okvqa_test_questions = json.load(f)
        okvqa_image_ids = set(['{:012d}'.format(int(item['image_id'])) for item in  okvqa_test_questions['questions']])

        f = open('/scratch/local/ssd/anjun/datasets/coco/karpathy/dataset_coco.json')
        karpathy = json.load(f)
        karpathy_image_ids = []
        for item in karpathy['images']:
            if item['split'] == 'test':
                karpathy_image_ids.append('{:012d}'.format(int(item['cocoid'])))
        union_image_ids = set(okvqa_image_ids).union(set(karpathy_image_ids))
        print(len(union_image_ids))
    
    os.makedirs(f'{cfg.OUTPUT_DIR}/images/val_adv/', exist_ok=True)
    os.makedirs(f'{cfg.OUTPUT_DIR}/images/val_adv_visuals/', exist_ok=True)

    with ExitStack() as stack:
        model.train()
        model.requires_grad_(False)

        start_data_time = time.perf_counter()
        dict.get(callbacks or {}, "on_start", lambda: None)()
        for idx, inputs in enumerate(data_loader):
            model.train()

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
            if args.filter_vqa_cap and fns[0].split('.')[0] not in union_image_ids:
                # print(fns[0].split('.')[0], 'not needed for karpathy test')
                continue

            clean_images = [x["image"].float().detach().clone().to(model.device) for x in inputs]
            perturbation = [torch.zeros_like(x["image"]).float().to(model.device) for x in inputs]
            unscaled_gts = [copy.deepcopy(x["instances"]) for x in inputs]
            for p in perturbation:
                p.requires_grad_(True) #torch.Size([3, H, W]) 
            adv_inputs = copy.deepcopy(inputs)
            
            attack_optimizer = torch.optim.AdamW(perturbation, lr=5e-1)

            for iter in range(250):
                for s, _ in enumerate(perturbation): 
                    ### Inject perturabation
                    adv_inputs[s]["image"] = (clean_images[s] + perturbation[s].clamp(-8., 8.)).clamp(0., 255.)

                    ### Rescale both clean and adv samples
                    ratio = torch.rand((1,)).to(adv_inputs[s]["image"].device)*0.5 + 0.75
                    target_scale = (int(ratio*ori_sizes[s][0]), int(ratio*ori_sizes[s][1]))
                    adv_inputs[s]["image"] = F.interpolate(adv_inputs[s]["image"].unsqueeze(0), target_scale, mode='bicubic').squeeze(0)
                    adv_inputs[s]["instances"].gt_masks = F.interpolate(unscaled_gts[s].gt_masks.float().unsqueeze(0), target_scale, mode='bicubic').squeeze(0).bool()

                loss_dict = model(adv_inputs)
                losses = -sum(loss_dict.values())
                
                attack_optimizer.zero_grad()
                losses.backward()
                attack_optimizer.step()
                
            dict.get(callbacks or {}, "after_inference", lambda: None)()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            # For evaluation purposes, use the target dataset ensembled prompts
            # rather than source dataset category names 
            # model.criterion = SetCriterion(...)
            for s, _ in enumerate(perturbation): 
                adv_inputs[s]["image"] = (clean_images[s] + perturbation[s].clamp(-8., 8.)).clamp(0., 255.)
                adv_inputs[s]["instances"] = unscaled_gts[s]
            model.eval()
            processed_results = model(adv_inputs)
            evaluator.process(inputs, processed_results)
            total_eval_time += time.perf_counter() - start_eval_time
            
            for i, x in enumerate(adv_inputs):
                images = x["image"].float().detach().unsqueeze(0).cpu()
                # images = images[:, :, :ori_sizes[i][0], :ori_sizes[i][1]]
                fn = x['file_name'].split('/')[-1]
                torch.save(images, f'{cfg.OUTPUT_DIR}/images/val_adv/{fn[:-4]}.pt')
                torchvision.utils.save_image(images/255., f'{cfg.OUTPUT_DIR}/images/val_adv_visuals/{fn}')
                print(f'{cfg.OUTPUT_DIR}/images/val_adv/{fn[:-4]}.pt')

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
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            if cfg.PSEUDO:
                evaluator = partial(
                    GeneralizedPseudoSemSegEvaluator,
                    with_prior=cfg.PSEUDO_WITH_PRIOR,
                    reject_threshold=cfg.PSEUDO_REJECT_THRESHOLD,
                )
            else:
                if cfg.MODEL.EVALUATIONTYPE.SEG_AP:
                    evaluator = GeneralizedSemSegEvaluatorClassAgnosticSeenUnseen
                    if cfg.MODEL.EVALUATIONTYPE.SEG_AP_OUTPUT:
                        evaluator = GeneralizedSemSegEvaluatorClassAgnosticOutput
                else:
                    evaluator = GeneralizedSemSegEvaluator
                
                # evaluator = GeneralizedSemSegEvaluator
            evaluator_list.append(
                evaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    post_process_func=dense_crf_post_process
                    if cfg.TEST.DENSE_CRF
                    else None,
                )
            )

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "classification":
            evaluator_list.append(ClassificationEvaluator(dataset_name))

        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
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
        dataset = None
        mapper = None
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_caption_semantic":
            mapper = MaskFormerCaptionSemanticDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_binary_semantic":
            mapper = MaskFormerBinarySemanticDatasetMapper(cfg, True)
            dataset = dataset_sample_per_class(cfg)
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if cfg.ORACLE:
            if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
                mapper = MaskFormerSemanticDatasetMapper(cfg, False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_caption_semantic":
                mapper = MaskFormerCaptionSemanticDatasetMapper(cfg, False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_binary_semantic":
                mapper = MaskFormerBinarySemanticDatasetMapper(cfg, False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "propsoal_classification":
                mapper = ProposalClasificationDatasetMapper(cfg, False)
            else:
                mapper = OracleDatasetMapper(cfg, False)
        else:
            # mapper = None
            if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
                mapper = MaskFormerSemanticDatasetMapper(cfg, False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_caption_semantic":
                mapper = MaskFormerCaptionSemanticDatasetMapper(cfg, False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_binary_semantic":
                mapper = MaskFormerBinarySemanticDatasetMapper(cfg, False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == "propsoal_classification":
                mapper = ProposalClasificationDatasetMapper(cfg, False)
            else:
                mapper = OracleDatasetMapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:
        ::
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Here the default print/log frequency of each writer is used.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            # WandbWriter(),
        ]

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
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
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
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
    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        # import pdb; pdb.set_trace()
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)
            # if param.requires_grad:
            #     print("grad "+name)
        # import pdb; pdb.set_trace()
        self._trainer._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former"
    )
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

        if cfg.TEST.AUG.ENABLED:
            res = Trainer.test_with_TTA(cfg, model)
        else:
            res = Trainer.test(cfg, model)
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
        "--filter-vqa-cap",
        action="store_true",
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
