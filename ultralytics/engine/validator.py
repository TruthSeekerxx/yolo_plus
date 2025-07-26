# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo mode=val model=yolo11n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolo11n.pt                 # PyTorch
                          yolo11n.torchscript        # TorchScript
                          yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolo11n_openvino_model     # OpenVINO
                          yolo11n.engine             # TensorRT
                          yolo11n.mlpackage          # CoreML (macOS-only)
                          yolo11n_saved_model        # TensorFlow SavedModel
                          yolo11n.pb                 # TensorFlow GraphDef
                          yolo11n.tflite             # TensorFlow Lite
                          yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolo11n_paddle_model       # PaddlePaddle
                          yolo11n.mnn                # MNN
                          yolo11n_ncnn_model         # NCNN
                          yolo11n_imx_model          # Sony IMX
                          yolo11n_rknn_model         # Rockchip RKNN
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils import ops
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.utils.metrics import ap_per_class, ConfusionMatrix

class BaseValidator:
    """
    A base class for creating validators.

    This class provides the foundation for validation processes, including model evaluation, metric computation, and
    result visualization.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary containing dataset information.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names mapping.
        seen (int): Number of images seen so far during validation.
        stats (dict): Statistics collected during validation.
        confusion_matrix: Confusion matrix for classification evaluation.
        nc (int): Number of classes.
        iouv (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (list): List to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
            batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.
        stride (int): Model stride for padding calculations.
        loss (torch.Tensor): Accumulated loss during training validation.

    Methods:
        __call__: Execute validation process, running inference on dataloader and computing performance metrics.
        match_predictions: Match predictions to ground truth objects using IoU.
        add_callback: Append the given callback to the specified event.
        run_callbacks: Run all callbacks associated with a specified event.
        get_dataloader: Get data loader from dataset path and batch size.
        build_dataset: Build dataset from image path.
        preprocess: Preprocess an input batch.
        postprocess: Postprocess the predictions.
        init_metrics: Initialize performance metrics for the YOLO model.
        update_metrics: Update metrics based on predictions and batch.
        finalize_metrics: Finalize and return all metrics.
        get_stats: Return statistics about the model's performance.
        print_results: Print the results of the model's predictions.
        get_desc: Get description of the YOLO model.
        on_plot: Register plots for visualization.
        plot_val_samples: Plot validation samples during training.
        plot_predictions: Plot YOLO model predictions on batch images.
        pred_to_json: Convert predictions to JSON format.
        eval_json: Evaluate and return JSON format of prediction statistics.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """
        Initialize a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            args (SimpleNamespace, optional): Configuration for the validator.
            _callbacks (dict, optional): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.nm = 0
        
        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.00 if self.args.task == "obb" else 0.000
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            # After self.update_metrics(preds, batch)
            self.save_predictions_with_extra(preds, batch, str(self.save_dir / "results.csv"))

            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """
        Match predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,).
            true_classes (torch.Tensor): Target class indices of shape (M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
            use_scipy (bool, optional): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        """Append the given callback to the specified event."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Run all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """Build dataset from image path."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocess an input batch."""
        print("HEREE")
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)
        return batch

    def postprocess(self, preds):
        """Postprocess the predictions."""
        print("HERE")
        print(preds)
        return ops.non_max_suppression(
            preds,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            classes=None,
            agnostic=False,
            max_det=self.args.max_det,
            nc=self.nc,
        )

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        # Handle AutoBackend or standard model
        if isinstance(model, AutoBackend):
            inner_model = model.model  # Access the inner torch model
        else:
            inner_model = de_parallel(model)
        
        # Access the detection head (assuming it's the last module)
        detect_head = inner_model.model[-1] if hasattr(inner_model, 'model') else inner_model
        
        self.nc = detect_head.nc
        self.names = detect_head.names if hasattr(detect_head, 'names') else inner_model.names
        self.nm = detect_head.extra_regs
        
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # IoU thresholds for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.stats = []  # List of per-image dicts: {'tp': tensor(N_preds, niou), 'conf': list, 'pred_cls': list, 'target_cls': list, 'extra_regs': list}
        self.seen = 0
        
        
    def save_predictions_with_extra(self, preds, batch, save_path):
        """
        Save predictions with extra regression outputs to CSV.
        Each row: x1, y1, x2, y2, conf, cls, reg1, reg2, reg3, reg4,...
        Handles both tensor and dict outputs.
        """
        with open(save_path, "a") as f:
            for si, pred in enumerate(preds):
                if pred is None or len(pred) == 0:
                    continue
                # Handle dict or tensor
                if isinstance(pred, dict):
                    # If 'det' key present, use it
                    if "det" in pred:
                        rows = pred["det"].cpu().numpy()
                    else:
                        # If there are other possible keys, handle as needed
                        continue
                else:
                    rows = pred.cpu().numpy()
                for row in rows:
                    line = ",".join([f"{x:.6f}" for x in row])
                    f.write(line + "\n")



    def update_metrics(self, preds, batch):
        """Update metrics based on predictions and batch."""
        n = len(preds)  # Number of images in batch
        for si, pred in enumerate(preds):  # Per image
            self.seen += 1
            idx = (batch["batch_idx"] == si).nonzero().squeeze(-1)  # Indices for this image
            if len(idx) == 0:  # No labels
                continue
    
            # Ground truth
            cls = batch["cls"][idx]
            bbox = batch["bboxes"][idx]
            nl = len(cls)
            tcls = cls.tolist() if nl else []
    
            # Predictions
            if pred is None or len(pred) == 0:
                if nl:
                    pass  # Metrics will reflect 0 recall
                continue
            predn = pred.clone()  # xyxy, conf, cls, reg1, reg2, reg3, reg4
    
            # Scale predictions from input scale to original shape
            input_h, input_w = batch["img"][si].shape[1:]  # C H W, so [1:] = H W
            ori_h, ori_w = batch["ori_shape"][si]
            predn[:, :4] = ops.scale_boxes((input_h, input_w), predn[:, :4], (ori_h, ori_w))
    
            # Extract extra regression values (if any)
            extra_regs = predn[:, 6:6+self.nm].cpu().tolist() if predn.shape[1] > 6 else []  # reg1, reg2, reg3, reg4
    
            # Scale ground truth from normalized (0-1) to original pixel scale
            scale = torch.tensor([ori_w, ori_h, ori_w, ori_h], device=self.device)
            tbox = ops.xywh2xyxy(bbox) * scale  # Convert to xyxy in original pixel
    
            if not nl:
                continue
    
            # Compute IoU matrix (N_preds x N_gt)
            iou = ops.bbox_iou(predn[:, :4], tbox, xyxy=True, CIoU=False)
    
            # Match predictions to GT
            pred_cls = predn[:, 5]  # Class index
            correct = self.match_predictions(pred_cls, cls, iou)
    
            # Confusion matrix update
            labels = torch.cat((cls.view(-1, 1), tbox), 1)  # cls xyxy
            detections = predn[:, :6]  # xyxy, conf, cls (ignore extra regs for confusion matrix)
            self.confusion_matrix.process_batch(detections, labels)
    
            # Append per-image stats
            detected_conf = predn[:, 4].cpu().tolist()
            detected_cls = pred_cls.cpu().tolist()
            self.stats.append({
                'tp': correct.cpu(),
                'conf': detected_conf,
                'pred_cls': detected_cls,
                'target_cls': tcls,
                'extra_regs': extra_regs  # Store extra regressions
            })
            
    def finalize_metrics(self):
        """Finalize and return all metrics."""
        self.metrics = self.get_stats()

    def get_stats(self):
        """Return statistics about the model's performance."""
        stats_list = self.stats
        if not stats_list:
            return {
                "mp": 0.0,
                "mr": 0.0,
                "map50": 0.0,
                "map": 0.0,
                "ap": np.zeros((0, self.niou)),
                "p": np.zeros(0),
                "r": np.zeros(0),
            }
    
        tp = [s['tp'] for s in stats_list]
        conf = [s['conf'] for s in stats_list]
        pred_cls = [s['pred_cls'] for s in stats_list]
        target_cls = [s['target_cls'] for s in stats_list]
    
        # Flatten lists
        conf_flat = [item for sublist in conf for item in sublist]
        pred_cls_flat = [item for sublist in pred_cls for item in sublist]
        target_cls_flat = [item for sublist in target_cls for item in sublist]
    
        # Convert to numpy arrays
        conf_flat = np.array(conf_flat) if conf_flat else np.zeros(0)
        pred_cls_flat = np.array(pred_cls_flat) if pred_cls_flat else np.zeros(0)
        target_cls_flat = np.array(target_cls_flat) if target_cls_flat else np.zeros(0)
    
        tp_flat = torch.cat(tp, 0).cpu().numpy() if tp else np.zeros((0, self.niou))
    
        # Compute number of targets per class
        self.nt = np.bincount(target_cls_flat.astype(int), minlength=self.nc)  # Store as self.nt
    
        # Use ap_per_class for standard computation
        tp, fp, p, r, ap, unique_classes = ap_per_class(
            tp=tp_flat,
            conf=conf_flat,
            pred_cls=pred_cls_flat,
            target_cls=target_cls_flat,
            plot=False,
            save_dir=self.save_dir,
            names=self.names
        )
    
        # Averages
        mp = p.mean() if len(p) else 0.0
        mr = r.mean() if len(r) else 0.0
        map50 = ap[:, 0].mean() if len(ap) else 0.0
        map = ap.mean() if len(ap) else 0.0
    
        stats = {
            "mp": mp,
            "mr": mr,
            "map50": map50,
            "map": map,
            "ap": ap,
            "p": p,
            "r": r,
        }
        return stats

    def print_results(self):
        """Print the results of the model's predictions."""
        pf = '%22s' + '%11s' * 6  # print format
        LOGGER.info(pf % ('all', self.seen, sum([len(s['target_cls']) for s in self.stats]), self.metrics['mp'], self.metrics['mr'], self.metrics['map50'], self.metrics['map']))
        for i in range(self.nc):
            LOGGER.info(pf % (self.names[i], self.seen, self.nt[i], self.metrics['p'][i], self.metrics['r'][i], self.metrics['ap'][i, 0], self.metrics['ap'][i].mean()))
    
    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Return the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Register plots for visualization."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def plot_val_samples(self, batch, ni):
        """Plot validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plot YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass