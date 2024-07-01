from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class FreezeArguments:
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    """

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            'help':
            ('The number of trainable layers for freeze (partial-parameter) fine-tuning. '
             'Positive numbers mean the last n layers are set as trainable, '
             'negative numbers mean the first n layers are set as trainable.')
        },
    )
    freeze_trainable_modules: str = field(
        default='all',
        metadata={
            'help':
            ('Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. '
             'Use commas to separate multiple modules. '
             'Use `all` to specify all the available modules. '
             'LLaMA choices: [`mlp`, `self_attn`], '
             'BLOOM & Falcon & ChatGLM choices: [`mlp`, `self_attention`], '
             'Qwen choices: [`mlp`, `attn`], '
             'InternLM2 choices: [`feed_forward`, `attention`], '
             'Others choices: the same as LLaMA.')
        },
    )
    freeze_extra_modules: Optional[str] = field(
        default=None,
        metadata={
            'help':
            ('Name(s) of modules apart from hidden layers to be set as trainable '
             'for freeze (partial-parameter) fine-tuning. '
             'Use commas to separate multiple modules.')
        },
    )


@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            'help':
            ('Name(s) of modules apart from LoRA layers to be set as trainable '
             'and saved in the final checkpoint. '
             'Use commas to separate multiple modules.')
        },
    )
    lora_rank: int = field(
        default=8,
        metadata={'help': 'The intrinsic dimension for LoRA fine-tuning.'},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'The scale factor for LoRA fine-tuning (default: lora_rank * 2).'
        },
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={'help': 'Dropout rate for the LoRA fine-tuning.'},
    )
    lora_target: str = field(
        default='all',
        metadata={
            'help':
            ('Name(s) of target modules to apply LoRA. '
             'Use commas to separate multiple modules. '
             'Use `all` to specify all the linear modules. '
             'LLaMA choices: [`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`], '
             'BLOOM & Falcon & ChatGLM choices: [`query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h`], '
             'Baichuan choices: [`W_pack`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`], '
             'Qwen choices: [`c_attn`, `attn.c_proj`, `w1`, `w2`, `mlp.c_proj`], '
             'InternLM2 choices: [`wqkv`, `wo`, `w1`, `w2`, `w3`], '
             'Others choices: the same as LLaMA.')
        },
    )
    lora_bias: Literal['none', 'all', 'lora_only'] = field(
        default='none',
        metadata={'help': 'Whether or not to use bias for LoRA layers.'},
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={'help': 'LoRA plus learning rate ratio (lr_B / lr_A).'},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={
            'help': 'LoRA plus learning rate for lora embedding layers.'
        },
    )
    use_qlora: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to use the LoRA+ method for learning rate scaling.'
        },
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to use the rank stabilization scaling factor for LoRA layer.'
        },
    )
    use_dora: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to use the weight-decomposed lora method (DoRA).'
        },
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to create a new adapter with randomly initialized weight.'
        },
    )


@dataclass
class QuantArguments:
    # 使用8-bit的adam，是否可以调整为LION或Sophia，甚至deepspeed还提供了多个1-bit优化器选择
    adam8bit: bool = field(default=False, metadata={'help': 'Use 8-bit adam.'})
    # 使用的位宽，默认为4。
    quant_bit: Optional[int] = field(
        default=4,
        metadata={
            'help':
            'The number of bits to quantize the model using bitsandbytes.'
        },
    )
    # 量化类型，可以选择`fp4`或`nf4`
    quant_type: Literal['fp4', 'nf4'] = field(
        default='nf4',
        metadata={
            'help': 'Quant data type to use. Should be one of `fp4` or `nf4`.'
        },
    )
    # 是否使用二次量化
    double_quant: bool = field(
        default=True,
        metadata={
            'help': 'Compress the quant statistics through double quant.'
        },
    )
    quant_device_map: Optional[Literal['auto']] = field(
        default=None,
        metadata={
            'help':
            'Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0.'
        },
    )
    export_quant_bit: Optional[int] = field(
        default=None,
        metadata={
            'help': 'The number of bits to quantize the exported model.'
        },
    )
    export_quant_dataset: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Path to the dataset or dataset name to use in quantizing the exported model.'
        },
    )
    export_quant_nsamples: int = field(
        default=128,
        metadata={'help': 'The number of samples used for quant.'},
    )
    export_quant_maxlen: int = field(
        default=1024,
        metadata={
            'help': 'The maximum length of the model inputs used for quant.'
        },
    )

    def __post_init__(self):
        if self.quant_bit is not None:
            assert self.quant_bit in [
                4,
                8,
            ], 'We only accept 4-bit or 8-bit quant.'
        if self.quant_type is not None:
            assert self.quant_type in [
                'nf4',
                'fp4',
            ], 'We only accept `nf4` or `fp4` quant type.'
        assert self.export_quant_bit in [
            None,
            8,
            4,
            3,
            2,
        ], 'We only accept 2/3/4/8-bit quantization.'

        if self.export_quant_bit is not None and self.export_quant_dataset is None:
            raise ValueError(
                'Quantization dataset is necessary for exporting.')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RLHFArguments:
    r"""
    Arguments pertaining to the PPO and DPO training.
    """

    pref_beta: float = field(
        default=0.1,
        metadata={'help': 'The beta parameter in the preference loss.'},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={
            'help':
            'The supervised fine-tuning loss coefficient in DPO training.'
        },
    )
    pref_loss: Literal['sigmoid', 'hinge', 'ipo', 'kto_pair', 'orpo',
                       'simpo'] = field(
                           default='sigmoid',
                           metadata={'help': 'The type of DPO loss to use.'},
                       )
    dpo_beta: float = field(
        default=0.1,
        metadata={'help': 'The beta parameter for the DPO loss.'},
    )
    dpo_loss: Literal['sigmoid', 'hinge', 'ipo', 'kto_pair'] = field(
        default='sigmoid',
        metadata={'help': 'The type of DPO loss to use.'},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={
            'help':
            'The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5.'
        },
    )
    dpo_ftx: float = field(
        default=0.0,
        metadata={
            'help':
            'The supervised fine-tuning loss coefficient in DPO training.'
        },
    )
    kto_beta: float = field(
        default=0.1,
        metadata={'help': 'The beta parameter for the KTO loss.'},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={
            'help':
            'The weight factor of the desirable losses in KTO training.'
        },
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={
            'help':
            'The weight factor of the undesirable losses in KTO training.'
        },
    )
    kto_ftx: float = field(
        default=0.0,
        metadata={
            'help':
            'The supervised fine-tuning loss coefficient in KTO training.'
        },
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={'help': 'The target reward margin term in SimPO loss.'},
    )
    orpo_beta: float = field(
        default=0.1,
        metadata={
            'help':
            'The beta (lambda) parameter in the ORPO loss representing the weight of the SFT loss.'
        },
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={
            'help':
            'The number of mini-batches to make experience buffer in a PPO optimization step.'
        },
    )
    ppo_epochs: int = field(
        default=4,
        metadata={
            'help':
            'The number of epochs to perform in a PPO optimization step.'
        },
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={'help': 'Use score normalization in PPO training.'},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={
            'help': 'Target KL value for adaptive KL control in PPO training.'
        },
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={
            'help':
            'Whiten the rewards before compute advantages in PPO training.'
        },
    )
    ref_model: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Path to the reference model used for the PPO or DPO training.'
        },
    )
    ref_model_adapters: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the adapters of the reference model.'},
    )
    ref_model_quant_bit: Optional[int] = field(
        default=None,
        metadata={
            'help': 'The number of bits to quantize the reference model.'
        },
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Path to the reward model used for the PPO training.'
        },
    )
    reward_model_adapters: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the adapters of the reward model.'},
    )
    reward_model_quant_bit: Optional[int] = field(
        default=None,
        metadata={'help': 'The number of bits to quantize the reward model.'},
    )
    reward_model_type: Literal['lora', 'full', 'api'] = field(
        default='lora',
        metadata={
            'help':
            'The type of the reward model in PPO training. Lora model only supports lora training.'
        },
    )


@dataclass
class GaloreArguments:
    r"""
    Arguments pertaining to the GaLore algorithm.
    """

    use_galore: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to use the gradient low-Rank projection (GaLore).'
        },
    )
    galore_target: str = field(
        default='all',
        metadata={
            'help':
            ('Name(s) of modules to apply GaLore. Use commas to separate multiple modules. '
             'Use `all` to specify all the linear modules.')
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={'help': 'The rank of GaLore gradients.'},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={'help': 'Number of steps to update the GaLore projection.'},
    )
    galore_scale: float = field(
        default=0.25,
        metadata={'help': 'GaLore scaling coefficient.'},
    )
    galore_proj_type: Literal['std', 'reverse_std', 'right', 'left',
                              'full'] = field(
                                  default='std',
                                  metadata={
                                      'help': 'Type of GaLore projection.'
                                  },
                              )
    galore_layerwise: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to enable layer-wise update to further save memory.'
        },
    )


@dataclass
class BAdamArgument:
    r"""
    Arguments pertaining to the BAdam optimizer.
    """

    use_badam: bool = field(
        default=False,
        metadata={'help': 'Whether or not to use the BAdam optimizer.'},
    )
    badam_mode: Literal['layer', 'ratio'] = field(
        default='layer',
        metadata={
            'help': 'Whether to use layer-wise or ratio-wise BAdam optimizer.'
        },
    )
    badam_start_block: Optional[int] = field(
        default=None,
        metadata={'help': 'The starting block index for layer-wise BAdam.'},
    )
    badam_switch_mode: Optional[Literal[
        'ascending', 'descending', 'random', 'fixed']] = field(
            default='ascending',
            metadata={
                'help':
                'the strategy of picking block to update for layer-wise BAdam.'
            },
        )
    badam_switch_interval: Optional[int] = field(
        default=50,
        metadata={
            'help':
            'Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update.'
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={'help': 'The ratio of the update for ratio-wise BAdam.'},
    )
    badam_mask_mode: Literal['adjacent', 'scatter'] = field(
        default='adjacent',
        metadata={
            'help':
            ('The mode of the mask for BAdam optimizer. '
             '`adjacent` means that the trainable parameters are adjacent to each other, '
             '`scatter` means that trainable parameters are randomly choosed from the weight.'
             )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            'help':
            ('The verbosity level of BAdam optimizer. '
             '0 for no print, 1 for print the block prefix, 2 for print trainable parameters.'
             )
        },
    )


@dataclass
class FinetuningArguments(
        FreezeArguments,
        LoraArguments,
        QuantArguments,
        RLHFArguments,
        GaloreArguments,
        BAdamArgument,
):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """

    pure_bf16: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to train model in purely bf16 precision (without AMP).'
        },
    )
    stage: Literal['pt', 'sft', 'rm', 'ppo', 'dpo', 'kto', 'orpo'] = field(
        default='sft',
        metadata={'help': 'Which stage will be performed in training.'},
    )
    finetuning_type: Literal['lora', 'freeze', 'full'] = field(
        default='lora',
        metadata={'help': 'Which fine-tuning method to use.'},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to make only the parameters in the expanded blocks trainable.'
        },
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={
            'help': 'Whether ot not to freeze vision tower in MLLM training.'
        },
    )
    train_mm_proj_only: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to train the multimodal projector for MLLM only.'
        },
    )
    plot_loss: bool = field(
        default=False,
        metadata={'help': 'Whether or not to save the training loss curves.'},
    )
    wandb_project: Optional[str] = field(
        default='llamatuner',
        metadata={'help': 'The name of the wandb project to log to.'},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={'help': 'The name of the wandb run.'},
    )

    def __post_init__(self):

        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(',')]
            return arg

        self.freeze_trainable_modules = split_arg(
            self.freeze_trainable_modules)
        self.freeze_extra_modules = split_arg(self.freeze_extra_modules)
        self.lora_alpha = self.lora_alpha or self.lora_rank * 2
        self.lora_target = split_arg(self.lora_target)
        self.additional_target = split_arg(self.additional_target)
        self.galore_target = split_arg(self.galore_target)

        assert self.finetuning_type in [
            'lora',
            'freeze',
            'full',
        ], 'Invalid fine-tuning method.'
        assert self.ref_model_quant_bit in [
            None,
            8,
            4,
        ], 'We only accept 4-bit or 8-bit quantization.'
        assert self.reward_model_quant_bit in [
            None,
            8,
            4,
        ], 'We only accept 4-bit or 8-bit quantization.'

        if self.stage == 'ppo' and self.reward_model is None:
            raise ValueError('`reward_model` is necessary for PPO training.')

        if (self.stage == 'ppo' and self.reward_model_type == 'lora'
                and self.finetuning_type != 'lora'):
            raise ValueError(
                '`reward_model_type` cannot be lora for Freeze/Full PPO training.'
            )

        if (self.stage == 'dpo' and self.dpo_loss != 'sigmoid'
                and self.dpo_label_smoothing > 1e-6):
            raise ValueError(
                '`dpo_label_smoothing` is only valid for sigmoid loss function.'
            )

        if self.use_llama_pro and self.finetuning_type == 'full':
            raise ValueError(
                '`use_llama_pro` is only valid for the Freeze or LoRA training.'
            )

        if self.use_galore and self.finetuning_type == 'lora':
            raise ValueError('Cannot use LoRA with GaLore together.')

        if self.use_galore and self.use_badam:
            raise ValueError('Cannot use GaLore with BAdam together.')

        if self.loraplus_lr_ratio is not None and self.finetuning_type != 'lora':
            raise ValueError(
                '`loraplus_lr_ratio` is only valid for the LoRA training.')
