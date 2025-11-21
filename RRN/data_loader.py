"""
DESCRIPTION:

    Data loading and preprocessing utilities.

    This module handles
        -   conversion from the proprietary reldata format to standard Python data structures (see data_structures.py),
        -   prepares data for training.
"""

# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #

# reldata
from reldata.io import kg_reader

import os
import sys
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

# own standard data structures
from data_structures import (
    Class,
    Relation,
    Individual,
    Membership,
    Triple,
    KnowledgeGraph,
    DataType,
)


# ---------------------------------------------------------------------------- #
#                                   FUNCTIONS                                  #
# ---------------------------------------------------------------------------- #


def convert_reldata_to_kg(reldata_kg) -> KnowledgeGraph:
    """
    Converts a reldata knowledge graph to standard Python data structures.

    Args:
        reldata_kg: Knowledge graph in proprietary reldata format

    Returns:
        KnowledgeGraph object with standard Python classes
    """

    # Convert classes
    classes = [
        Class(
            # If cls in classes._data has no name attribute, use a default name
            index=i,
            name=cls.name if hasattr(cls, "name") else f"Class_{i}",
        )
        # Loop over index and class in reldata_kg.classes._data
        for i, cls in enumerate(reldata_kg.classes._data)
    ]

    # Convert relations
    relations = [
        Relation(index=i, name=rel.name if hasattr(rel, "name") else f"Relation_{i}")
        for i, rel in enumerate(reldata_kg.relations._data)
    ]

    # Convert individuals (first pass - without memberships)
    individuals = []
    for ind in reldata_kg.individuals:
        individual = Individual(
            index=ind.index,
            name=ind.name if hasattr(ind, "name") else f"Individual_{ind.index}",
            classes=[],  # Will be populated below
        )
        individuals.append(individual)

    # Add memberships to individuals
    for ind_idx, reldata_ind in enumerate(reldata_kg.individuals):
        memberships = []
        for membership in reldata_ind.classes:
            cls = classes[membership.cls.index]
            memberships.append(
                Membership(
                    cls=cls,
                    is_member=membership.is_member,
                    is_inferred=membership.inferred,
                )
            )
        individuals[ind_idx].classes = memberships

    # Convert triples
    triples = []
    for triple in reldata_kg.triples:
        triples.append(
            Triple(
                subject=individuals[triple.subject.index],
                predicate=relations[triple.predicate.index],
                object=individuals[triple.object.index],
                positive=triple.positive,
                is_inferred=triple.inferred,
            )
        )

    return KnowledgeGraph(
        classes=classes, relations=relations, individuals=individuals, triples=triples
    )


def load_knowledge_graphs(
    input_dir: str, num_workers: int = None
) -> List[KnowledgeGraph]:
    """
    Loads knowledge graphs from a directory and converts them to standard format.

    Args:
        input_dir: Directory containing knowledge graph files

    Returns:
        List of KnowledgeGraph objects
    """
    print("Reading knowledge graphs from directory:", input_dir, file=sys.stderr)

    # Decide worker count; default to CPU count - 1 (>=1)
    workers = (
        num_workers if num_workers is not None else max(1, (os.cpu_count() or 1) - 1)
    )

    # Use the existing executor architecture in KgReader.read_all
    try:
        if workers > 1:
            print(
                f"Using {workers} workers to read knowledge graphs...", file=sys.stdout
            )
            with ThreadPoolExecutor(max_workers=workers) as ex:
                reldata_kgs = kg_reader.KgReader.read_all(
                    input_dir=input_dir, executor=ex
                )
        else:
            reldata_kgs = kg_reader.KgReader.read_all(input_dir=input_dir)
    except AttributeError as e:
        print(f"Error reading knowledge graphs: {e}", file=sys.stderr)
        sys.exit(1)

    print("Converting knowledge graphs to standard format...", file=sys.stderr)

    return [convert_reldata_to_kg(kg) for kg in reldata_kgs]


def preprocess_knowledge_graph(
    kg: KnowledgeGraph,
) -> dict[str, List]:
    """
    Preprocesses a knowledge graph for training

    Args:
        kg:         Knowledge graph to preprocess

    Returns:
        Dictionary containing:
        - List of   message passing     triples                                 (base facts = specified)
        - List of   message passing     membership vectors \in {-1,0,1}^|C|     (base facts = specified)
        - List of   TEST                triples                                 (inferred)
        - List of   TEST                membership vectors                      (inferred)
        - List of   ALL=target          triples                                 (specified AND inferred)
        - List of   ALL=target          membership vectors                      (specified AND inferred)
    """

    # Base fact triples and all triples
    base_triples = [t for t in kg.triples if not t.is_inferred]
    inferred_triples = [t for t in kg.triples if t.is_inferred]
    all_triples = kg.triples

    # Base fact memberships and all memberships
    base_memberships = []
    inferred_memberships = []
    all_memberships = []

    # Non-factual memberships are memberships that are not known facts,
    # i.e., they are not explicitly stated in the knowledge graph.

    for individual in kg.individuals:
        # Initialize vectors with zeros
        base_membership_vec = [0] * len(kg.classes)
        inferred_membership_vec = [0] * len(kg.classes)
        all_membership_vec = [0] * len(kg.classes)

        # Populate based on class memberships
        for membership in individual.classes:
            class_idx = membership.cls.index

            # Membership value: 1 if member, -1 if not member, 0 if unknown
            #
            # -> based on indicator function
            #           1_KB : individuals(KB) -> {-1,0,1}^|C|
            #           1_KB(i) = ( 1 if i is member of C
            #                      -1 if i is not member of C
            #                       0 if otherwise )
            #
            # -> see page 7 in the RRN paper
            #
            membership_value = 1 if membership.is_member else -1

            # Set in all membership vector
            all_membership_vec[class_idx] = membership_value

            # Only set in base membership vector
            if not membership.is_inferred:
                base_membership_vec[class_idx] = membership_value
            else:
                inferred_membership_vec[class_idx] = membership_value

        # Note that 1_KB(i) = 0 for all classes C where the membership is unknown
        # (i.e., not explicitly stated in individual.classes)

        base_memberships.append(base_membership_vec)
        inferred_memberships.append(inferred_membership_vec)
        all_memberships.append(all_membership_vec)

    return {
        "base_triples": base_triples,
        "base_memberships": base_memberships,
        "inferred_triples": inferred_triples,
        "inferred_memberships": inferred_memberships,
        "all_triples": all_triples,
        "all_memberships": all_memberships,
    }


def custom_collate_fn(
    batch: List[dict[str, List]],
) -> dict[str, List]:
    """
    Custom collate function for DataLoader.

    A collate function merges a list of samples to form a mini-batch of Tensor(s).

    Since each knowledge graph is processed individually, this function simply returns the first
    (and only) element of the batch.

    Args:
        batch:  Batch of preprocessed knowledge graphs (list of knowledge graph dictionaries)
                List of dictionaries with:
                -   base_triples
                -   base_memberships
                -   inferred_triples
                -   inferred_memberships
                -   all_triples
                -   all_memberships

    Returns:
        Single (first) preprocessed knowledge graph dictionary
    """
    return batch[0]
