"""Command-line entry point for the RedDebate framework.

This module parses CLI arguments, prints the resolved configuration, and
dispatches to :func:`redDebate.run.run_debate`. It supports configuring
debater models, optional roles (devil/angel, Socratic questioner, evaluator,
feedback generator, self-critic), the dataset(s) to iterate over, textual memory
backends (array-based long-term memory or a Pinecone vector store), and
PEFT/LoRA fine-tuning as continues long-term memory.
"""

from redDebate.run import run_debate
import argparse

def parse_debate_args():
    """Parse command-line arguments for a debate run.

    Returns:
        argparse.Namespace: parsed arguments. The ``--models`` and
        ``--datasets`` entries are lists of ``<type>:<name>[:use_chat]`` and
        ``<dataset_name>:<dataset_path>`` strings respectively (see argparse
        ``--help`` for the full per-argument descriptions).
    """
    parser = argparse.ArgumentParser(description="Facilitate interaction between different types of LLMs.")

    parser.add_argument(
        "--models",
        nargs='+',
        type=str,
        required=False,
        default=list(),
        help="List of debating models to use, in the format <type>:<model_name_or_path><use_chat>. For example, 'openai:gpt-4o-mini:true' or 'huggingface:/path/to/local/model:false'.",
    )

    parser.add_argument(
        "--angel_model",
        type=str,
        required=False,
        default=None,
        help="Specify the angel model in the format <type>:<model_name_or_path><use_chat>. For example, 'openai:gpt-4o-mini:true' or 'huggingface:/path/to/local/model:false'.",
    )

    parser.add_argument(
        "--devil_model",
        type=str,
        required=False,
        default=None,
        help="Specify the devil model in the format <type>:<model_name_or_path><use_chat>. For example, 'openai:gpt-4o-mini:true' or 'huggingface:/path/to/local/model:false'.",
    )

    parser.add_argument(
        "--evaluator",
        type=str,
        required=False,
        default=None,
        help="Specify the evaluator model in the format <type>:<model_name_or_path><use_chat>. For example, 'openai:gpt-4o-mini:true' or 'huggingface:/path/to/local/model:false'. You can use openai:moderation to use OpenAI's moderation API.",
    )

    parser.add_argument(
        "--feedback_generator",
        type=str,
        required=False,
        default=None,
        help="Specify the feedback generator model in the format <type>:<model_name_or_path><use_chat>. For example, 'openai:gpt-4o-mini:true' or 'huggingface:/path/to/local/model:false'.",
    )

    parser.add_argument(
        "--questioner_model",
        type=str,
        required=False,
        default=None,
        help="Specify the questioner model in the format <type>:<model_name_or_path><use_chat>. For example, 'openai:gpt-4o-mini:true' or 'huggingface:/path/to/local/model:false'.",
    )

    parser.add_argument(
        "--self_critique_model",
        type=str,
        required=False,
        default=None,
        help="Specify the self-critique model in the format <type>:<model_name_or_path><use_chat>. For example, 'openai:gpt-4o-mini:true' or 'huggingface:/path/to/local/model:false'.",
    )

    parser.add_argument(
        "--datasets",
        nargs='+',
        type=str,
        required=True,
        help="List of datasets to use for each model interaction in the format <dataset_name>:<dataset_path>. For example, 'harmbench:/path/to/local/model'. Choose dataset name from: 'harmbench', 'cosafe', 'toxicchat', 'hhrlhf', or 'saferdialogues'.",
    )

    parser.add_argument(
        "--debate_rounds",
        type=int,
        default=2,
        help="Number of debate rounds between the models. Default is 2.",
    )

    parser.add_argument(
        "--max_total_debates",
        type=int,
        default=None,
        help="Maximum number of debates to run. If not set, there is no constraint.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="debate.log",
        help="File to log the debate. Default is 'debate.log'.",
    )

    parser.add_argument(
        "--textual_memory_index",
        type=str,
        default=None,
        help="Name of the index in vector database to store long-term memory. Default is None which initiate the array-based memory instead.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save and load checkpoints. Default is None.",
    )

    parser.add_argument(
        "--llamaguard_cuda_device",
        type=int,
        default=0,
        help="CUDA device index for LlamaGuard. Default is 0.",
    )

    parser.add_argument(
        "--peft_memory",
        action="store_true",
        default=False,
        help=(
            "Enable PEFT/LoRA fine-tuning as a form of long-term memory. "
            "The debating HuggingFace models are fine-tuned on accumulated feedback every "
            "--train_steps debates. Long-term memory continues to work independently if a "
            "--feedback_generator is provided. Default is disabled."
        ),
    )

    parser.add_argument(
        "--peft_directory",
        type=str,
        default=None,
        help=(
            "Sub-directory (relative to HF_HUB_CACHE) where LoRA checkpoints are saved. "
            "Required when --peft_memory is set. To run inference with a previously trained LoRA "
            "model without further training, simply pass its merged checkpoint path in --models."
        ),
    )

    parser.add_argument(
        "--train_steps",
        type=int,
        default=10,
        help="Number of feedback samples to accumulate before triggering PEFT training. Default is 10.",
    )

    parser.add_argument(
        "--human_in_the_loop",
        type=int,
        default=0,
        help="Number of human participants to include in each debate round. Each human is prompted for input via stdin. Default is 0 (no humans).",
    )

    # Parse arguments
    args = parser.parse_args()

    print("Configurations:")
    print(f"Debating Models: {args.models}")
    print(f"Angel Model: {args.angel_model}")
    print(f"Devil Model: {args.devil_model}")
    print(f"Evaluator Model: {args.evaluator}")
    print(f"Feedback Generator Model: {args.feedback_generator}")
    print(f"Questioner Model: {args.questioner_model}")
    print(f"Self Critique Model: {args.self_critique_model}")
    print(f"Datasets: {args.datasets}")
    print(f"Debate Rounds: {args.debate_rounds}")
    print(f"Max Total Debates: {args.max_total_debates}")
    print(f"Output File: {args.output_file}")
    print(f"Textual Memory Index: {args.textual_memory_index}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")
    print(f"LlamaGuard CUDA Device: {args.llamaguard_cuda_device}")
    print(f"PEFT Memory: {args.peft_memory}")
    print(f"PEFT Directory: {args.peft_directory}")
    print(f"PEFT Train Steps: {args.train_steps}")
    print(f"Humans in the Loop: {args.human_in_the_loop}")

    return args

if __name__ == "__main__":
    args = parse_debate_args()
    run_debate(
        debater_models=args.models,
        angel_model=args.angel_model,
        devil_model=args.devil_model,
        evaluator_model=args.evaluator,
        feedback_generator=args.feedback_generator,
        questioner_model=args.questioner_model,
        self_critique_model=args.self_critique_model,
        datasets=args.datasets,
        debate_rounds=args.debate_rounds,
        max_total_debates=args.max_total_debates,
        output_file=args.output_file,
        textual_memory_index=args.textual_memory_index,
        llamaguard_cuda_device=args.llamaguard_cuda_device,
        checkpoint_dir=args.checkpoint_dir,
        peft_memory=args.peft_memory,
        peft_directory=args.peft_directory,
        train_steps=args.train_steps,
        human_in_the_loop=args.human_in_the_loop,
    )