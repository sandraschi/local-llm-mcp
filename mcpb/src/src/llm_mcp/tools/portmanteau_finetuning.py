"""LLM Fine-tuning portmanteau tool for Local LLM MCP server.

This tool consolidates all fine-tuning operations (LoRA, Sparse, DoRA) into a single interface
following the portmanteau pattern.
"""

from typing import Any

from llm_mcp.tools.dora_tools import (
    dora_list_models_impl,
    dora_load_model_impl,
    dora_prepare_for_training_impl,
    dora_train_impl,
    dora_unload_model_impl,
)
from llm_mcp.tools.lora_tools import (
    _lora_list_adapters_impl,
    _lora_list_loaded_impl,
    _lora_load_adapter_impl,
    _lora_unload_adapter_impl,
)
from llm_mcp.tools.sparse_tools import (
    sparse_list_models_impl,
    sparse_load_model_impl,
    sparse_prepare_for_training_impl,
    sparse_train_impl,
    sparse_unload_model_impl,
)
from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Import FastMCP components
try:
    from fastmcp import FastMCP
    from fastmcp.tools import Tool
    FASTMCP_AVAILABLE = True
except ImportError:
    logger.error("FastMCP not available - portmanteau tools require FastMCP >= 2.12.0")
    FASTMCP_AVAILABLE = False


async def llm_finetuning(
    operation: str,
    # Model loading parameters
    model_name: str | None = None,
    model_id: str | None = None,
    # LoRA parameters
    adapter_dir: str | None = None,
    adapter_name: str | None = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
    # Sparse parameters
    sparsity_ratio: float = 0.5,
    sparsity_type: str = "unstructured",
    # Common parameters
    use_4bit: bool = True,
    use_double_quant: bool = True,
    quant_type: str = "nf4",
    compute_dtype: str = "bfloat16",
    device_map: str = "auto",
    # Training parameters
    train_dataset: Any | None = None,
    output_dir: str = "./output",
    learning_rate: float = 0.0002,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 3,
    max_steps: int = -1,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    optim: str = "paged_adamw_32bit",
    lr_scheduler_type: str = "cosine",
    max_grad_norm: float = 0.3,
    logging_steps: int = 10,
    save_steps: int = 200,
    save_total_limit: int = 3,
    report_to: str | None = None,
    use_gradient_checkpointing: bool = True,
    use_flash_attention_2: bool = True,
    use_cpu_offload: bool = False,
) -> dict[str, Any]:
    """Comprehensive fine-tuning tool for Local LLM MCP server.

    PORTMANTEAU PATTERN: Consolidates 15+ fine-tuning operations into one tool.

    SUPPORTED OPERATIONS:

    LoRA Operations:
    - lora_list_adapters: List available LoRA adapters (optional adapter_dir)
    - lora_load_adapter: Load LoRA adapter (requires adapter_name)
    - lora_unload_adapter: Unload LoRA adapter (requires adapter_name)
    - lora_list_loaded: List currently loaded LoRA adapters

    Sparse Operations:
    - sparse_load_model: Load model for sparse fine-tuning (requires model_name)
    - sparse_prepare_training: Prepare sparse model for training (requires model_id)
    - sparse_train: Train sparse model (requires model_id, train_dataset)
    - sparse_unload_model: Unload sparse model (requires model_id)
    - sparse_list_models: List loaded sparse models

    DoRA Operations:
    - dora_load_model: Load model with DoRA (requires model_name)
    - dora_prepare_training: Prepare DoRA model for training (requires model_id)
    - dora_train: Train DoRA model (requires model_id, train_dataset)
    - dora_unload_model: Unload DoRA model (requires model_id)
    - dora_list_models: List loaded DoRA models

    Args:
        operation: Operation to perform (see SUPPORTED OPERATIONS above)
        model_name: Model name/path for loading operations
        model_id: Model ID for training/unloading operations
        adapter_dir: Directory for LoRA adapters
        adapter_name: Adapter name for LoRA operations
        lora_rank: LoRA rank (default: 8)
        lora_alpha: LoRA alpha scaling (default: 16)
        lora_dropout: LoRA dropout (default: 0.1)
        target_modules: Target modules for LoRA (optional)
        sparsity_ratio: Sparsity ratio for sparse training (default: 0.5)
        sparsity_type: Sparsity type (default: "unstructured")
        use_4bit: Use 4-bit quantization (default: True)
        use_double_quant: Use double quantization (default: True)
        quant_type: Quantization type (default: "nf4")
        compute_dtype: Compute dtype (default: "bfloat16")
        device_map: Device mapping (default: "auto")
        train_dataset: Training dataset for training operations
        output_dir: Output directory (default: "./output")
        learning_rate: Learning rate (default: 0.0002)
        batch_size: Batch size per device (default: 4)
        gradient_accumulation_steps: Gradient accumulation steps (default: 4)
        num_train_epochs: Number of epochs (default: 3)
        max_steps: Maximum steps (-1 for all, default: -1)
        warmup_ratio: Warmup ratio (default: 0.03)
        weight_decay: Weight decay (default: 0.01)
        optim: Optimizer (default: "paged_adamw_32bit")
        lr_scheduler_type: LR scheduler (default: "cosine")
        max_grad_norm: Max gradient norm (default: 0.3)
        logging_steps: Logging frequency (default: 10)
        save_steps: Save frequency (default: 200)
        save_total_limit: Max checkpoints (default: 3)
        report_to: Reporting integration (optional)
        use_gradient_checkpointing: Use gradient checkpointing (default: True)
        use_flash_attention_2: Use Flash Attention 2 (default: True)
        use_cpu_offload: Use CPU offload (default: False)

    Returns:
        Operation-specific result dictionary
    """
    try:
        # LoRA operations
        if operation == "lora_list_adapters":
            return await _lora_list_adapters_impl(adapter_dir)

        elif operation == "lora_load_adapter":
            if not adapter_name:
                return {"error": "adapter_name required for lora_load_adapter operation"}
            return await _lora_load_adapter_impl(adapter_name, adapter_dir)

        elif operation == "lora_unload_adapter":
            if not adapter_name:
                return {"error": "adapter_name required for lora_unload_adapter operation"}
            return await _lora_unload_adapter_impl(adapter_name)

        elif operation == "lora_list_loaded":
            return await _lora_list_loaded_impl()

        # Sparse operations
        elif operation == "sparse_load_model":
            if not model_name:
                return {"error": "model_name required for sparse_load_model operation"}
            return await sparse_load_model_impl(
                model_name=model_name,
                model_id=model_id,
                sparsity_ratio=sparsity_ratio,
                sparsity_type=sparsity_type,
                use_4bit=use_4bit,
                use_double_quant=use_double_quant,
                quant_type=quant_type,
                compute_dtype=compute_dtype,
                device_map=device_map,
            )

        elif operation == "sparse_prepare_training":
            if not model_id:
                return {"error": "model_id required for sparse_prepare_training operation"}
            return await sparse_prepare_for_training_impl(
                model_id=model_id,
                output_dir=output_dir,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_train_epochs,
                max_steps=max_steps,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                optim=optim,
                lr_scheduler_type=lr_scheduler_type,
                max_grad_norm=max_grad_norm,
                logging_steps=logging_steps,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                report_to=report_to,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_flash_attention_2=use_flash_attention_2,
                use_cpu_offload=use_cpu_offload,
            )

        elif operation == "sparse_train":
            if not model_id or not train_dataset:
                return {"error": "model_id and train_dataset required for sparse_train operation"}
            return await sparse_train_impl(
                model_id=model_id,
                train_dataset=train_dataset,
                eval_dataset=None,  # Not implemented in current sparse_train_impl
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_steps=int(warmup_ratio * 100),  # Convert ratio to steps
                logging_steps=logging_steps,
                save_steps=save_steps,
                eval_steps=None,
                output_dir=output_dir,
                save_total_limit=save_total_limit,
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False,
                fp16=(compute_dtype == "float16"),
                bf16=(compute_dtype == "bfloat16"),
                max_grad_norm=max_grad_norm,
                group_by_length=True,
                report_to=report_to,
            )

        elif operation == "sparse_unload_model":
            if not model_id:
                return {"error": "model_id required for sparse_unload_model operation"}
            return await sparse_unload_model_impl(model_id)

        elif operation == "sparse_list_models":
            return await sparse_list_models_impl()

        # DoRA operations
        elif operation == "dora_load_model":
            if not model_name:
                return {"error": "model_name required for dora_load_model operation"}
            return await dora_load_model_impl(
                model_name=model_name,
                model_id=model_id,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                use_double_quant=use_double_quant,
                quant_type=quant_type,
                compute_dtype=compute_dtype,
                device_map=device_map,
            )

        elif operation == "dora_prepare_training":
            if not model_id:
                return {"error": "model_id required for dora_prepare_training operation"}
            return await dora_prepare_for_training_impl(
                model_id=model_id,
                output_dir=output_dir,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_train_epochs,
                max_steps=max_steps,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                optim=optim,
                lr_scheduler_type=lr_scheduler_type,
                max_grad_norm=max_grad_norm,
                logging_steps=logging_steps,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                report_to=report_to,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_flash_attention_2=use_flash_attention_2,
                use_cpu_offload=use_cpu_offload,
            )

        elif operation == "dora_train":
            if not model_id or not train_dataset:
                return {"error": "model_id and train_dataset required for dora_train operation"}
            return await dora_train_impl(
                model_id=model_id,
                train_dataset=train_dataset,
                eval_dataset=None,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_steps=int(warmup_ratio * 100),
                logging_steps=logging_steps,
                save_steps=save_steps,
                eval_steps=None,
                output_dir=output_dir,
                save_total_limit=save_total_limit,
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False,
                fp16=(compute_dtype == "float16"),
                bf16=(compute_dtype == "bfloat16"),
                max_grad_norm=max_grad_norm,
                group_by_length=True,
                report_to=report_to,
            )

        elif operation == "dora_unload_model":
            if not model_id:
                return {"error": "model_id required for dora_unload_model operation"}
            return await dora_unload_model_impl(model_id)

        elif operation == "dora_list_models":
            return await dora_list_models_impl()

        else:
            lora_ops = ["lora_list_adapters", "lora_load_adapter", "lora_unload_adapter", "lora_list_loaded"]
            sparse_ops = ["sparse_load_model", "sparse_prepare_training", "sparse_train", "sparse_unload_model", "sparse_list_models"]
            dora_ops = ["dora_load_model", "dora_prepare_training", "dora_train", "dora_unload_model", "dora_list_models"]
            all_ops = lora_ops + sparse_ops + dora_ops

            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": all_ops,
                "categories": {
                    "lora": lora_ops,
                    "sparse": sparse_ops,
                    "dora": dora_ops
                }
            }

    except Exception as e:
        logger.error(f"Error in llm_finetuning operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {e!s}", "operation": operation}


def register_llm_finetuning_tools(mcp):
    """Register the LLM Fine-tuning portmanteau tool with the MCP server."""
    if not FASTMCP_AVAILABLE:
        logger.error("Cannot register LLM Fine-tuning tools - FastMCP not available")
        return mcp

    @mcp.tool()
    async def llm_finetuning_tool(
        operation: str,
        model_name: str | None = None,
        model_id: str | None = None,
        adapter_dir: str | None = None,
        adapter_name: str | None = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: list[str] | None = None,
        sparsity_ratio: float = 0.5,
        sparsity_type: str = "unstructured",
        use_4bit: bool = True,
        use_double_quant: bool = True,
        quant_type: str = "nf4",
        compute_dtype: str = "bfloat16",
        device_map: str = "auto",
        train_dataset: Any | None = None,
        output_dir: str = "./output",
        learning_rate: float = 0.0002,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 3,
        max_steps: int = -1,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        optim: str = "paged_adamw_32bit",
        lr_scheduler_type: str = "cosine",
        max_grad_norm: float = 0.3,
        logging_steps: int = 10,
        save_steps: int = 200,
        save_total_limit: int = 3,
        report_to: str | None = None,
        use_gradient_checkpointing: bool = True,
        use_flash_attention_2: bool = True,
        use_cpu_offload: bool = False,
    ) -> dict[str, Any]:
        """LLM Fine-tuning Portmanteau Tool - Consolidated fine-tuning operations.

        This tool consolidates all fine-tuning operations (LoRA, Sparse, DoRA) into a single interface,
        reducing the number of MCP tools while maintaining full functionality.

        Use the 'operation' parameter to specify what you want to do:

        LoRA Operations:
        - lora_list_adapters: List available LoRA adapters
        - lora_load_adapter: Load a LoRA adapter
        - lora_unload_adapter: Unload a LoRA adapter
        - lora_list_loaded: List loaded LoRA adapters

        Sparse Fine-tuning:
        - sparse_load_model: Load model for sparse fine-tuning
        - sparse_prepare_training: Prepare sparse model for training
        - sparse_train: Train sparse model
        - sparse_unload_model: Unload sparse model
        - sparse_list_models: List loaded sparse models

        DoRA Fine-tuning:
        - dora_load_model: Load model with DoRA
        - dora_prepare_training: Prepare DoRA model for training
        - dora_train: Train DoRA model
        - dora_unload_model: Unload DoRA model
        - dora_list_models: List loaded DoRA models
        """
        return await llm_finetuning(
            operation=operation,
            model_name=model_name,
            model_id=model_id,
            adapter_dir=adapter_dir,
            adapter_name=adapter_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            sparsity_ratio=sparsity_ratio,
            sparsity_type=sparsity_type,
            use_4bit=use_4bit,
            use_double_quant=use_double_quant,
            quant_type=quant_type,
            compute_dtype=compute_dtype,
            device_map=device_map,
            train_dataset=train_dataset,
            output_dir=output_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            optim=optim,
            lr_scheduler_type=lr_scheduler_type,
            max_grad_norm=max_grad_norm,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            report_to=report_to,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_flash_attention_2=use_flash_attention_2,
            use_cpu_offload=use_cpu_offload,
        )

    logger.info("Registered LLM Fine-tuning portmanteau tool")
    return mcp
