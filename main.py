from redDebate.run import run_debate
import argparse

def parse_debate_args():
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
        "--long_term_memory_index_name",
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
    print(f"Long-term Memory Index Name: {args.long_term_memory_index_name}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")

    return args

if __name__ == "__main__":
    args = parse_debate_args()
    run_debate(
        args.models,
        args.angel_model,
        args.devil_model,
        args.evaluator,
        args.feedback_generator,
        args.questioner_model,
        args.self_critique_model,
        args.datasets,
        args.debate_rounds,
        args.max_total_debates,
        args.output_file,
        args.long_term_memory_index_name,
        args.checkpoint_dir,
    )