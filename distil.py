#!/usr/bin/env python3
# 3/7/2024
# Copyright (C) 2024 Charles O. Goddard

from typing import Optional
import torch
import transformers
import datasets
import click
import multiprocessing
import wandb
from transformers.utils import is_apex_available
from llama_rope_offset import llama_patch_rope_offset, llama_set_rope_offset


if is_apex_available():
    from apex import amp


class DistillationTrainer(transformers.Trainer):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        teacher: transformers.PreTrainedModel,
        sequence_length: int,
        temperature: float = 2.0,
        kl_weight: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(model=model, *args, **kwargs)
        self.teacher = teacher
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self.loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.rope_offset = 0
        self.sequence_length = sequence_length
        self.max_sequence_length = model.config.max_position_embeddings

    def compute_loss(self, model, inputs, return_outputs=False, split=False):
        student_output = model(**inputs)

        with torch.no_grad():
            teacher_output = self.teacher(**inputs)

        distillation_loss = (
            self.loss_fct(
                torch.nn.functional.log_softmax(
                    student_output.logits / self.temperature, dim=-1
                ),
                torch.nn.functional.softmax(
                    teacher_output.logits / self.temperature, dim=-1
                ).to(device=student_output.logits.device),
            )
            * self.temperature**2
        )

        student_loss = student_output.loss
        loss = (1 - self.kl_weight) * student_loss + self.kl_weight * distillation_loss

        if split:
            return (loss, student_loss, distillation_loss)
        return (loss, student_output) if return_outputs else loss

    def on_step_begin(
        self,
        _args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        model: transformers.PreTrainedModel,
        **kwargs,
    ):
        new_offset = (self.sequence_length // 2 * state.global_step) % (
            self.max_sequence_length - self.sequence_length
        )
        self.rope_offset = new_offset
        llama_set_rope_offset(model, self.rope_offset)
        return control

    def on_step_end(self, model, **kwargs):
        llama_set_rope_offset(model, 0)  # in case of evaluation

    def training_step(self, model, inputs):
        model.train()

        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss, student_loss, distillation_loss = self.compute_loss(
                model, inputs, split=True
            )

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            student_loss = student_loss.mean()
            distillation_loss = distillation_loss.mean()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        self.log(
            {
                "train_student_loss": student_loss.item(),
                "train_distillation_loss": distillation_loss.item(),
            }
        )

        return loss.detach() / self.args.gradient_accumulation_steps


@click.command("distil")
@click.option("--model", "-m", type=str, required=True)
@click.option("--dataset", "-d", type=str, required=True)
@click.option("--batch-size", type=int, default=8)
@click.option("--sequence-length", type=int, default=512)
@click.option("--teacher", "-t", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.option("--device", type=str, default="auto")
@click.option("--teacher-device", type=str, default="auto")
@click.option("--num-workers", type=int, default=None)
@click.option("--temperature", type=float, default=2.0)
@click.option("--kl-weight", type=float, default=0.5)
@click.option("--lr", type=float, default=5e-5)
@click.option("--lr-scheduler", type=str, default="inverse_sqrt")
@click.option("--warmup-steps", type=int, default=20)
@click.option("--eval-steps", type=int, default=1000)
@click.option("--save-steps", type=int, default=1000)
@click.option("--logging-steps", type=int, default=1)
@click.option("--epochs", type=int, default=1)
@click.option("--gradient-accumulation-steps", "-G", type=int, default=1)
@click.option("--max-grad-norm", type=float, default=1.0)
@click.option("--trust-remote-code/--no-trust-remote-code", is_flag=True, default=False)
@click.option(
    "--teacher-precision",
    type=click.Choice(["f32", "f16", "bf16", "int8", "int4"]),
    default=None,
)
@click.option("--seed", type=int, default=42)
@click.option("--save-total-limit", type=int, default=5)
@click.option("--rope-offset/--no-rope-offset", is_flag=True, default=False)
@click.option("--project", type=str, default=None)
@click.option("--resume-from", type=str, default=None)
def main(
    model: str,
    dataset: str,
    batch_size: int,
    sequence_length: int,
    teacher: str,
    output: str,
    device: str,
    teacher_device: str,
    num_workers: Optional[int],
    temperature: float,
    kl_weight: float,
    lr: float,
    lr_scheduler: str,
    warmup_steps: int,
    eval_steps: int,
    save_steps: int,
    logging_steps: int,
    epochs: int,
    trust_remote_code: bool,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    teacher_precision: Optional[str],
    seed: int,
    save_total_limit: int,
    rope_offset: bool,
    project: Optional[str],
    resume_from: Optional[str],
):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    if project:
        wandb.init(project=project)

    student = transformers.AutoModelForCausalLM.from_pretrained(
        model,
        device_map=device,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if rope_offset:
        llama_patch_rope_offset(
            student,
            max_position_embeddings=student.config.max_position_embeddings,
        )

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_kwargs = {
        "device_map": teacher_device,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if teacher_precision == "int4":
        teacher_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif teacher_precision == "int8":
        teacher_kwargs["load_in_8bit"] = True
    elif teacher_precision == "bf16":
        teacher_kwargs["torch_dtype"] = torch.bfloat16
    elif teacher_precision == "f16":
        teacher_kwargs["torch_dtype"] = torch.float16
    elif teacher_precision == "f32":
        teacher_kwargs["torch_dtype"] = torch.float32

    teacher = transformers.AutoModelForCausalLM.from_pretrained(
        teacher, **teacher_kwargs
    )

    train_ds, eval_ds = load_data(dataset, sequence_length, num_workers, tokenizer)

    train_args = transformers.TrainingArguments(
        output_dir=output,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        report_to="wandb",
        logging_dir=output,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=max_grad_norm,
        optim="adamw_bnb_8bit",
        seed=seed,
        bf16=True,
        save_total_limit=save_total_limit,
        torch_compile=False,
        load_best_model_at_end=True,
        resume_from_checkpoint=resume_from,
    )

    trainer = DistillationTrainer(
        model=student,
        teacher=teacher,
        temperature=temperature,
        kl_weight=kl_weight,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model()


def load_data(
    dataset: str,
    sequence_length: int,
    num_workers: int,
    tokenizer: transformers.PreTrainedTokenizerBase,
):
    ds = datasets.load_dataset(dataset)

    def _tokenize(ds):
        if "input_ids" in ds.column_names:
            return ds
        return ds.map(
            lambda x: tokenizer(
                x["text"],
                padding="max_length",
                truncation=True,
                max_length=sequence_length,
            ),
            batched=True,
            num_proc=num_workers,
        )

    def _label(ds):
        if "labels" in ds.column_names:
            return ds
        return ds.map(
            lambda x: {"labels": x["input_ids"]},
            batched=True,
            num_proc=num_workers,
        )

    if isinstance(ds, datasets.DatasetDict):
        train_ds = ds["train"]
        eval_ds = ds.get("eval", None)
    else:
        train_ds = ds
        eval_ds = None

    train_ds = _label(_tokenize(train_ds))
    if eval_ds is not None:
        eval_ds = _label(_tokenize(eval_ds))

    return train_ds, eval_ds


if __name__ == "__main__":
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    main()
