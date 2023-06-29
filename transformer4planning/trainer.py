from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import  nested_detach
from transformers.trainer_callback import TrainerState, TrainerControl, IntervalStrategy, DefaultFlowCallback
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import logging

class CustomCallback(DefaultFlowCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        # control.should_log = False
        if args.logging_strategy == IntervalStrategy.EPOCH and state.epoch % args.eval_interval == 0:
            control.should_log = True

        # Evaluate
        # control.should_evaluate = False
        if args.evaluation_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch and state.epoch % args.eval_interval == 0:
            control.should_evaluate = True

        # Save
        # control.should_save = False
        if args.save_strategy == IntervalStrategy.EPOCH and state.epoch % args.eval_interval == 0:
            control.should_save = True

        return control

@dataclass
class PlanningTrainingArguments(TrainingArguments):
    eval_interval: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "how many epoch the model performan evaluation."
            )
        },
    )

class PlanningTrainer(Trainer):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        # loss_without_labels = True if len(self.label_names) == 0 and return_loss else False
        loss_without_labels = True

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        # if has_labels or loss_without_labels:
        #     labels = nested_detach(tuple(inputs.get(name) is not None for name in self.label_names))
        #     if len(labels) == 1:
        #         labels = labels[0]
        # else:
        #     labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # # TODO: this needs to be fixed and made cleaner later.
                    # if self.args.past_index >= 0:
                    #     self._past = outputs[self.args.past_index - 1]
        batch_generate_eval = True
        if model.mode == 'OA-OA':
            batch_generate_eval = False
            print('batch generate for OA-OA is not implemented yet')
        if self.model.model_args.k not in [-1]:
            batch_generate_eval = False
            print('batch generate for TopK is not implemented yet')

        if batch_generate_eval:
            # run generate for sequence of actions
            # TODO: support different frame interval, change sequence length from 40
            if self.model.model_args.autoregressive:
                prediction_actions_in_batch = self.model.generate(**inputs)
                prediction_trajectory_in_batch = self.model.compute_normalized_points(prediction_actions_in_batch)
                actions_label_in_batch = inputs["trajectory"]
                trajectory_label_in_batch = self.model.compute_normalized_points(actions_label_in_batch)
            else:
                pass
            # compute ade
            x_error = prediction_trajectory_in_batch[:, :, 0] - trajectory_label_in_batch[:, -40:, 0]
            y_error = prediction_trajectory_in_batch[:, :, 1] - trajectory_label_in_batch[:, -40:, 1]
            ade = torch.sqrt(x_error ** 2 + y_error ** 2)
            ade = ade.mean()
            self.ade = (ade + self.ade * self.eval_itr)/(self.eval_itr + 1)
            self.ade = float(self.ade)  # tensor to float to save in json
            # compute fde
            x_error = prediction_trajectory_in_batch[:, -1, 0] - trajectory_label_in_batch[:, -1, 0]
            y_error = prediction_trajectory_in_batch[:, -1, 1] - trajectory_label_in_batch[:, -1, 1]
            fde = torch.sqrt(x_error ** 2 + y_error ** 2)
            fde = fde.mean()
            self.fde = (fde + self.fde * self.eval_itr)/(self.eval_itr + 1)
            self.fde = float(self.fde)  # tensor to float to save in json

        self.eval_itr += 1
        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, None)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        self.fde = 0
        self.ade = 0
        self.eval_itr = 0
        eval_output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        result = dict()
        if self.model.model_args.autoregressive and self.model.clf_metrics is not None:
            # run classsification metrics
            result["accuracy"] = self.model.clf_metrics["accuracy"].compute()
            result["f1"] = self.model.clf_metrics["f1"].compute(average="macro")
            result["precision"] = self.model.clf_metrics["precision"].compute(average="macro")
            result["recall"] = self.model.clf_metrics["recall"].compute(average="macro")
        result["ade"] = float(self.ade)
        result["fde"] = float(self.fde)
        logging.info("***** Eval results *****")
        logging.info(f"  {result}")
        self.log(result)
        return eval_output
