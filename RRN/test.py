"""
DESCRIPTION

    Test script of the revised Recursive Reasoning Network (RRN) implementation
    with batched relation updates, which allows for maximal GPU utilization.

AUTHOR

    Vincent Van Schependom

NOTATION

    *               matrix multiplication
    \cdot           dot product
    [;]             vector concatenation
    (a x b x c)     tensor of shape (a, b, c)

    d               embedding size
    N               number of RRN iterations
    C               |classes(KB)|, i.e. number of classes
    R               |relations(KB)|, i.e. number of relations
    M               number of individuals in the KB
    Br              batch size in RRN relation updates (number of triples for a specific predicate, either positive or negative)
    Bt              batch size in training (here: Bt = M = number of individuals in the KB, since 1 batch = 1 KB)
"""

# PyTorch
import torch
import torch.nn as nn

# Standard libraries
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import sys
import os

# Own modules
from data_structures import KnowledgeGraph, Triple, DataType
from rrn_model_batched import RRN, ClassesMLP, RelationMLP
from data_loader import load_knowledge_graphs, preprocess_knowledge_graph
from device import get_device
from tt_common import initialize_model
from globals import EMBEDDING_SIZE, ITERATIONS, VERBOSE, DATA_TYPE


def load_model_checkpoint(
    model: nn.Module, checkpoint_path: str, device: torch.device
) -> None:
    """
    Loads model weights from checkpoint.

    Args:
        model           : PyTorch model to load weights into
        checkpoint_path : Path to checkpoint file
        device          : Device to load model on
    """
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )


def get_latest_checkpoint_epoch(checkpoint_dir: Path) -> int:
    """
    Finds the latest checkpoint epoch in the given directory.
    This is handy if all subdirectories in the checkpoint/ directory are created based on the job date,
    like I did on the DTU HPC cluster.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Latest epoch number found in checkpoint filenames
    """

    # Scan directory for checkpoint files
    max_epoch = -1
    for file in checkpoint_dir.iterdir():
        if file.suffix == ".pth":
            parts = file.stem.split("_epoch-")
            if len(parts) == 2 and parts[1].isdigit():
                epoch = int(parts[1])
                if epoch > max_epoch:
                    max_epoch = epoch

    # No checkpoints found
    if max_epoch == -1:
        raise FileNotFoundError(
            f"No checkpoint files found in directory: {checkpoint_dir}"
        )

    return max_epoch


def load_trained_model(
    checkpoint_dir: str,
    checkpoint_epoch: int,
    reference_kg: KnowledgeGraph,
    embedding_size: int,
    iterations: int,
    device: torch.device,
) -> Tuple[RRN, nn.ModuleList]:
    """
    Loads a trained RRN model and its associated MLPs from checkpoints.

    This function reconstructs the model architecture based on a reference
    knowledge graph (which provides the ontology structure) and then loads
    the trained weights.

    Args:
        checkpoint_dir  : Directory containing checkpoint files
        checkpoint_epoch: Epoch number of checkpoint to load
        reference_kg    : Knowledge graph with the same ontology structure
        embedding_size  : Dimensionality of entity embeddings
        iterations      : Number of RRN message-passing iterations
        device          : Device to load models on

    Returns:
        Tuple of (RRN model, ModuleList of MLPs)
    """

    # Check if checkpoint number is provided
    if checkpoint_epoch is None:
        exit("Checkpoint epoch must be specified to load the model.")

    # ------------------------------ INITIALIZE RRN ------------------------------ #

    rrn, mlps = initialize_model(
        embedding_size=embedding_size,
        iterations=iterations,
        reference_kg=reference_kg,
        device=device,
    )

    # ---------------------------- RESTORE RNN WEIGHTS --------------------------- #

    # if checkpoint_epoch is None, use the latest checkpoint
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_epoch is None:
        checkpoint_epoch = get_latest_checkpoint_epoch(checkpoint_path)

    print(f"Loading checkpoint from epoch {checkpoint_epoch}...")

    rrn_checkpoint = checkpoint_path / f"rrn_epoch-{checkpoint_epoch}.pth"
    if not rrn_checkpoint.exists():
        raise FileNotFoundError(f"RRN checkpoint not found: {rrn_checkpoint}")
    load_model_checkpoint(rrn, str(rrn_checkpoint), device)

    # Load MLP weights
    for mlp_idx, mlp in enumerate(mlps):
        mlp_checkpoint = checkpoint_path / f"mlp{mlp_idx}_epoch-{checkpoint_epoch}.pth"
        if not mlp_checkpoint.exists():
            raise FileNotFoundError(f"MLP checkpoint not found: {mlp_checkpoint}")
        load_model_checkpoint(mlp, str(mlp_checkpoint), device)

    return rrn, mlps


def evaluate_classes(
    mlp: ClassesMLP,
    embeddings: torch.Tensor,
    membership_labels: List[List[int]],
    device: torch.device,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluates class membership predictions.

    This version processes all individuals at once and correctly
    masks unknown labels (0.5) from accuracy calculations.

    Args:
        mlp                 : Class prediction MLP
        embeddings          : Individual embeddings from RRN   (num_individuals, embedding_dim)
        membership_labels   : Ground truth membership labels   (list of lists of -1, 0, 1)
        device              : Device to run evaluation on

    Returns:
        Tuple of (accuracies, all_known_scores, positive_scores, negative_scores)
    """

    # Convert {-1, 0, 1} labels to {0.0, 0.5, 1.0}
    # where 0.5 = unknown, 1.0 = positive, 0.0 = negative
    #
    # member_of     <-> label = -1.0 <-> target = 1.0
    # ~member_of    <-> label = 0.0  <-> target = 0.0
    cls_targets = (
        torch.as_tensor(membership_labels, dtype=torch.float32, device=device) + 1
    ) / 2

    # Forward pass to get logits (num_individuals, num_classes)
    cls_logits = mlp(embeddings)

    # Convert logits to probabilities
    cls_probs = torch.sigmoid(cls_logits)

    # Generate actual predictions (rounded probabilities)
    cls_pred = cls_probs.round()
    # sigmoid(MLP_logit) = P(<s, member_of, C_i> | KB)
    # P(<s, ~member_of, C_i> | KB) = 1 - P(<s, member_of, C_i> | KB)
    #
    # So, all in all, this means that:
    #   round(sigmoid(MLP_logit)) = 1.0 <-> predicted member_of
    #   round(sigmoid(MLP_logit)) = 0.0 <-> predicted ~member_of

    # Calculate scores
    #   -> True/False where prediction matches label or not
    scores = (cls_pred == cls_targets).float()
    # Still (num_individuals, num_classes) shape

    # Create masks for known positive, known negative, and all known labels
    pos_mask = cls_targets == 1.0
    neg_mask = cls_targets == 0.0

    # Mask to filter out the "unknown" (0.5) labels
    known_mask = cls_targets != 0.5

    # Get the scores for each category (True/False tensors)
    positive_scores = scores[pos_mask]
    negative_scores = scores[neg_mask]
    all_known_scores = scores[known_mask]

    # Compute mean accuracies, handling empty tensors
    class_accuracies = {
        "all": all_known_scores.mean().item()
        if all_known_scores.numel() > 0
        else float("nan"),
        "positive": positive_scores.mean().item()
        if positive_scores.numel() > 0
        else float("nan"),
        "negative": negative_scores.mean().item()
        if negative_scores.numel() > 0
        else float("nan"),
    }

    return class_accuracies


def evaluate_triples(
    mlps: nn.ModuleList,
    embeddings: torch.Tensor,
    triples: List[Triple],
    device: torch.device,
) -> Tuple[Dict[str, float], List[torch.Tensor]]:
    """
    Evaluates relation predictions.

    Args:
        mlps      : List of MLPs (first one is for classes, rest for relations)
        embeddings: Entity embeddings from RRN (num_individuals, embedding_dim)
        triples   : List of relation triples to evaluate
        device    : Device to run evaluation on

    Returns:
        Tuple of (accuracy, list of per-triple scores, positive scores, negative scores)
    """
    triple_scores = []
    positive_scores = []
    negative_scores = []

    # TODO calculate all pred_logits at once for efficiency

    for triple in triples:
        pred_idx = triple.predicate.index
        subj_idx = triple.subject.index
        obj_idx = triple.object.index

        # Get MLP logits for this triple, where
        #   sigmoid(MLP_logit) = P(<s, R, o> | KB)
        #   P(<s, ~R, o> | KB) = 1 - P(<s, R, o> | KB)
        pred_logit = mlps[pred_idx + 1](embeddings[subj_idx, :], embeddings[obj_idx, :])

        # Set target based on whether triple is positive or negative
        if triple.positive:
            target = torch.tensor([1.0], dtype=torch.float32, device=device)
        else:
            target = torch.tensor([0.0], dtype=torch.float32, device=device)

        # sigmoid(pred_logit) = P(<s, R, o> | KB)
        prediction = torch.sigmoid(pred_logit)

        # Score = 1.0 if prediction matches target, else 0.0
        score = (prediction.round() == target).float()

        # Collect scores
        triple_scores.append(score)
        if triple.positive:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    # Compute overall accuracy, handling cases with no facts
    triple_accuracies = {
        "all": torch.cat(triple_scores).mean().item()
        if triple_scores
        else float("nan"),
        "positive": torch.cat(positive_scores).mean().item()
        if positive_scores
        else float("nan"),
        "negative": torch.cat(negative_scores).mean().item()
        if negative_scores
        else float("nan"),
    }

    return triple_accuracies


def test_on_knowledge_graph(
    rrn: RRN,
    mlps: nn.ModuleList,
    test_kg: KnowledgeGraph,
    device: torch.device,
    verbose: bool = True,
    data_type: DataType = DataType.ALL,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Tests the model, trained on an ontology Σ, on a single knowledge graph D_i.

    Args:
        rrn    : Trained RRN model
        mlps   : Trained MLP classifiers
        test_kg: Knowledge graph to test on (must share ontology with training)
        device : Device to run evaluation on
        verbose: Whether to print detailed progress

    Returns:
        Tuple of (class_accuracies, triple_accuracies),
        both dictionaries with 'all', 'positive', 'negative' accuracies
    """

    # Set to evaluation mode
    rrn.eval()
    mlps.eval()

    # Preprocess the test knowledge graph
    triples, membership_labels = preprocess_knowledge_graph(
        test_kg, data_type=data_type
    )

    if verbose:
        print("Test KG Statistics:")
        print(f"  Individuals: {len(test_kg.individuals)}")
        print(f"  Triples ({data_type.name}):    {len(triples)}")
        print(
            f"        Of which positive triples: {sum(1 for t in triples if t.positive)}"
        )
        print(
            f"        Of which negative triples: {sum(1 for t in triples if not t.positive)}"
        )
        print(f"  Memberships ({data_type.name}): {len(membership_labels)}")
        print(
            f"      Of which positive memberships: {sum(sum(1 for v in labels if v == 1) for labels in membership_labels)}"
        )
        print(
            f"      Of which negative memberships: {sum(sum(1 for v in labels if v == -1) for labels in membership_labels)}"
        )

    # ------------------------------- RUN INFERENCE ------------------------------ #

    # Disable gradient calculations for evaluation
    with torch.no_grad():
        # Generate embeddings using RRN
        embeddings = rrn(triples, membership_labels).to(device)

        # Evaluate class membership predictions
        class_accuracies = evaluate_classes(
            mlps[0], embeddings, membership_labels, device
        )

        # Evaluate relation predictions
        triple_accuracies = evaluate_triples(mlps, embeddings, triples, device)

    return class_accuracies, triple_accuracies


def test_model(
    checkpoint_dir: str,
    checkpoint_epoch: int,
    test_data_dir: str,
    test_indices: Optional[List[int]] = None,
    embedding_size: int = 100,
    iterations: int = 7,
    verbose: bool = True,
    data_type: DataType = DataType.ALL,
) -> dict:
    """
    Main testing function that loads a trained model and evaluates it on test data.

    This function:
    1. Loads the trained RRN and MLPs from checkpoints
    2. Loads test knowledge graphs from the specified directory
    3. Evaluates the model on the test graphs
    4. Returns detailed results

    Args:
        checkpoint_dir  : Subdirectory containing model checkpoints
        checkpoint_epoch: Epoch number of checkpoint to load
        test_data_dir   : Directory containing test knowledge graphs
        test_indices    : Optional list of specific KG indices to test on (if None, tests on all)
        embedding_size  : Dimensionality of entity embeddings
        iterations      : Number of RRN message-passing iterations
        verbose         : Whether to print detailed progress
        data_type       : Type of data to use for testing (inferred, specific, all)

    Returns:
        Dictionary containing test results and statistics
    """

    # Select device
    device = get_device()

    if verbose:
        print(f"Testing on device: {device}\n")
        print("=" * 70)

    # ------------------------------ LOAD TEST DATA ------------------------------ #

    if verbose:
        print("Loading test knowledge graphs...")

    test_kgs = load_knowledge_graphs(test_data_dir)

    # Select specific KGs if indices provided,
    # otherwise use all of them.
    if test_indices is not None:
        test_kgs = [test_kgs[i] for i in test_indices]
        if verbose:
            print(f"Selected {len(test_kgs)} knowledge graphs for testing")

    if len(test_kgs) == 0:
        raise ValueError("No test knowledge graphs found!")

    # -------------------------------- LOAD MODEL -------------------------------- #
    # -> use first KG to define structure
    # -> create models (RRN + MLPs)
    # -> load weights from checkpoints

    # if checkpoint_epoch is None, use the latest checkpoint
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_epoch is None:
        checkpoint_epoch = get_latest_checkpoint_epoch(checkpoint_path)

    # Use first KG to determine ontology structure
    reference_kg = test_kgs[0]

    if verbose:
        print("\nOntology Structure:")
        print(f"  Classes: {len(reference_kg.classes)}")
        print(f"  Relations: {len(reference_kg.relations)}")

    # Load trained model
    if verbose:
        print(f"\nLoading model from epoch {checkpoint_epoch}...")

    rrn, mlps = load_trained_model(
        checkpoint_dir=checkpoint_dir,
        checkpoint_epoch=checkpoint_epoch,
        reference_kg=reference_kg,
        embedding_size=embedding_size,
        iterations=iterations,
        device=device,
    )

    if verbose:
        print("Model loaded successfully!")

    # ------------------------------ START EVALUATION ------------------------------ #

    # Test on all knowledge graphs
    results = {
        # Classes
        "all_class_accuracies": [],
        "negative_class_accuracies": [],
        "positive_class_accuracies": [],
        # Triples
        "all_triple_accuracies": [],
        "positive_triple_accuracies": [],
        "negative_triple_accuracies": [],
        # KB stats
        "num_individuals": [],
        "num_triples": [],
    }

    if verbose:
        print("\n" + "=" * 70)
        print("Running Evaluation")
        print("=" * 70)

    for idx, test_kg in enumerate(test_kgs):
        if verbose:
            print(f"\nTesting on KG {idx + 1}/{len(test_kgs)}...")

        class_accuracies, triple_accuracies = test_on_knowledge_graph(
            rrn=rrn,
            mlps=mlps,
            test_kg=test_kg,
            device=device,
            verbose=verbose,
            data_type=data_type,
        )

        # Overall accuracies
        results["all_class_accuracies"].append(class_accuracies["all"])
        results["positive_class_accuracies"].append(class_accuracies["positive"])
        results["negative_class_accuracies"].append(class_accuracies["negative"])
        results["all_triple_accuracies"].append(triple_accuracies["all"])
        results["positive_triple_accuracies"].append(triple_accuracies["positive"])
        results["negative_triple_accuracies"].append(triple_accuracies["negative"])

        # KB stats
        results["num_individuals"].append(len(test_kg.individuals))
        results["num_triples"].append(len(test_kg.triples))

        if verbose:
            print(
                f"  Class Accuracy:  {class_accuracies['all']:.4f} ({class_accuracies['all'] * 100:.2f}%)"
            )
            print(
                f"    Positive:      {class_accuracies['positive']:.4f} ({class_accuracies['positive'] * 100:.2f}%)"
            )
            print(
                f"    Negative:      {class_accuracies['negative']:.4f} ({class_accuracies['negative'] * 100:.2f}%)"
            )
            print(
                f"  Triple Accuracy: {triple_accuracies['all']:.4f} ({triple_accuracies['all'] * 100:.2f}%)"
            )
            print(
                f"    Positive:      {triple_accuracies['positive']:.4f} ({triple_accuracies['positive'] * 100:.2f}%)"
            )
            print(
                f"    Negative:      {triple_accuracies['negative']:.4f} ({triple_accuracies['negative'] * 100:.2f}%)"
            )

    # --------------------------- COMPUTE SUMMARY STATS -------------------------- #

    results["mean_class_accuracy"] = sum(results["all_class_accuracies"]) / len(
        results["all_class_accuracies"]
    )
    results["mean_positive_class_accuracy"] = sum(
        results["positive_class_accuracies"]
    ) / len(results["positive_class_accuracies"])
    results["mean_negative_class_accuracy"] = sum(
        results["negative_class_accuracies"]
    ) / len(results["negative_class_accuracies"])
    results["mean_triple_accuracy"] = sum(results["all_triple_accuracies"]) / len(
        results["all_triple_accuracies"]
    )
    results["mean_positive_triple_accuracy"] = sum(
        results["positive_triple_accuracies"]
    ) / len(results["positive_triple_accuracies"])
    results["mean_negative_triple_accuracy"] = sum(
        results["negative_triple_accuracies"]
    ) / len(results["negative_triple_accuracies"])

    results["min_class_accuracy"] = min(results["all_class_accuracies"])
    results["min_positive_class_accuracy"] = min(results["positive_class_accuracies"])
    results["min_negative_class_accuracy"] = min(results["negative_class_accuracies"])
    results["min_triple_accuracy"] = min(results["all_triple_accuracies"])
    results["min_positive_triple_accuracy"] = min(results["positive_triple_accuracies"])
    results["min_negative_triple_accuracy"] = min(results["negative_triple_accuracies"])
    results["max_class_accuracy"] = max(results["all_class_accuracies"])
    results["max_positive_class_accuracy"] = max(results["positive_class_accuracies"])
    results["max_negative_class_accuracy"] = max(results["negative_class_accuracies"])
    results["max_triple_accuracy"] = max(results["all_triple_accuracies"])
    results["max_positive_triple_accuracy"] = max(results["positive_triple_accuracies"])
    results["max_negative_triple_accuracy"] = max(results["negative_triple_accuracies"])

    if verbose:
        print("\n" + "=" * 70)
        print("Summary Statistics")
        print("=" * 70)
        print(f"Number of test KGs:        {len(test_kgs)}")

        print("\nClass Membership Predictions:")
        print(
            f"  Mean Accuracy:           {results['mean_class_accuracy']:.4f} ({results['mean_class_accuracy'] * 100:.2f}%)"
        )
        print(
            f"  Accuracy Range:          {results['min_class_accuracy']:.4f} - {results['max_class_accuracy']:.4f}"
        )
        print(
            f"  Mean Positive Accuracy:  {results['mean_positive_class_accuracy']:.4f} ({results['mean_positive_class_accuracy'] * 100:.2f}%)"
        )
        print(
            f"  Positive Accuracy Range:   {results['min_positive_class_accuracy']:.4f} - {results['max_positive_class_accuracy']:.4f}"
        )
        print(
            f"  Mean Negative Accuracy:  {results['mean_negative_class_accuracy']:.4f} ({results['mean_negative_class_accuracy'] * 100:.2f}%)"
        )
        print(
            f"  Negative Accuracy Range:   {results['min_negative_class_accuracy']:.4f} - {results['max_negative_class_accuracy']:.4f}"
        )

        print("\nRelation Predictions:")
        print(
            f"  Mean Accuracy:           {results['mean_triple_accuracy']:.4f} ({results['mean_triple_accuracy'] * 100:.2f}%)"
        )
        print(
            f"  Accuracy Range:          {results['min_triple_accuracy']:.4f} - {results['max_triple_accuracy']:.4f}"
        )
        print(
            f"  Mean Positive Accuracy:  {results['mean_positive_triple_accuracy']:.4f} ({results['mean_positive_triple_accuracy'] * 100:.2f}%)"
        )
        print(
            f"  Positive Accuracy Range:   {results['min_positive_triple_accuracy']:.4f} - {results['max_positive_triple_accuracy']:.4f}"
        )
        print(
            f"  Mean Negative Accuracy:  {results['mean_negative_triple_accuracy']:.4f} ({results['mean_negative_triple_accuracy'] * 100:.2f}%)"
        )
        print(
            f"  Negative Accuracy Range:   {results['min_negative_triple_accuracy']:.4f} - {results['max_negative_triple_accuracy']:.4f}"
        )
        print("=" * 70)

    return results


# ---------------------------------------------------------------------------- #
#                               MAIN ENTRY POINT                               #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ----------------------- CHECKPOINT AND DATA DIRECTORY ---------------------- #

    # First argument is checkpoint dir
    if len(sys.argv) < 2:
        print("Usage: python test.py <checkpoint_dir>")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]

    print("\n" + "=" * 70)
    print(f"Checkpoint Directory: {checkpoint_dir}")

    # Get the base data directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data/family-tree/out/test-20")

    # print to std error
    print(f"Data directory: {data_dir}")

    # --------------------------------- TEST ON N GRAPHS -------------------------------- #

    print("\n\n" + "=" * 70)
    print("Testing on multiple knowledge graphs")
    print("=" * 70 + "\n")

    results = test_model(
        checkpoint_dir=checkpoint_dir,
        checkpoint_epoch=None,  # Use latest checkpoint
        test_data_dir=data_dir,
        test_indices=None,  # Test on all KGs
        embedding_size=EMBEDDING_SIZE,
        iterations=ITERATIONS,
        verbose=VERBOSE,
        data_type=DATA_TYPE,
    )

    # write results to a file
    with open("test_results.txt", "w") as f:
        f.write("Test Results:\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
