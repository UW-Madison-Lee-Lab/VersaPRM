from math import log, sqrt
from typing import Any, Callable, Literal, Optional

import torch

from interfaces import LOGGING, BaseGenerator, BasePRM, ReasoningNode


class MCTS:
    """
    ### Args:
        generator (BaseGenerator): Generator LLM
        prm (BasePRM): PRM
        branching_factor (int): Branching factor of the tree
        n_iters (int): Number of MCTS iterations
        max_depth (int): Maximum depth of the tree
        selection_objective (Optional[Callable[[ReasoningNode], float]]): Selection function

    ### Algorithm:

    Overview ...

    1. selection

    2. expansion

    3. simulation

    4. backpropagation

    TODO: Cite algorithm
    """

    def __init__(
        self,
        generator: BaseGenerator,
        prm: BasePRM,
        branching_factor: int = 2,
        n_iters: int = 10,
        max_depth: int = 10,
        selection_objective: Optional[Callable[[ReasoningNode], float]] = None,
        score_aggregation: Literal["min", "mean", "last"] = "last",
    ) -> None:
        self.generator = generator
        self.prm = prm
        self.branching_factor = branching_factor
        self.n_iters = n_iters
        self.max_depth = max_depth
        self.score_aggregation = score_aggregation

        if selection_objective is None:

            def default_selection_objective(node: ReasoningNode) -> float:
                c = sqrt(2)
                if node.parent is None:  # root node
                    return node.score + c * sqrt(log(node.n_visits) / node.n_visits)
                else:
                    return node.score + c * sqrt(
                        log(node.parent.n_visits) / node.n_visits
                    )

            self.selection_objective = default_selection_objective
        else:
            self.selection_objective = selection_objective  # type: ignore

    def selection(
        self,
        node: ReasoningNode,
        depth: int = 0,
    ) -> tuple[ReasoningNode, int, bool]:
        if depth == self.max_depth:
            return node, depth, False

        if len(node.children) == 0:
            return node, depth, True

        assert len(node.children) == self.branching_factor
        sorted_children = sorted(
            node.children,
            key=self.selection_objective,
            reverse=True,
        )

        for child in sorted_children:
            if child.is_leaf:
                continue

            selected_node, selected_node_depth, is_valid = self.selection(
                child,
                depth + 1,
            )

            if is_valid:
                return selected_node, selected_node_depth, is_valid

        return node, depth, False

    def __call__(self, question: str) -> list[dict[str, Any]]:
        input_ids = self.generator.encode(question)
        gen_state = self.generator.init_state(input_ids)
        prm_state = self.prm.init_state(question)

        root = ReasoningNode(
            parent=None,
            children=[],
            current_ids=input_ids,
            gen_state=gen_state,
            prm_state=prm_state,
        )
        complete_paths: list[tuple[str, float, list[float]]] = []
        outputs: list[dict[str, Any]] = []

        for i in range(self.n_iters):
            if LOGGING:
                print(f"Generation round {i+1}")

            # selection
            node_selected, depth, is_valid = self.selection(root)
            if not is_valid:
                print("MCTS terminated early")
                break

            # expansion and simulation
            while not node_selected.is_leaf and depth < self.max_depth:
                input_ids = node_selected.current_ids
                gen_state = node_selected.gen_state
                assert node_selected.prm_state is not None
                prm_state = node_selected.prm_state
                prm_scores = node_selected.prm_scores

                input_ids = input_ids.repeat(self.branching_factor, 1)
                gen_state = self.generator.inflate_state(
                    gen_state,
                    self.branching_factor,
                )
                prm_state = self.prm.inflate_state(prm_state, self.branching_factor)

                gen_output_ids, gen_state = self.generator(input_ids, gen_state)
                is_complete = self.generator.is_complete(gen_output_ids)

                new_input_ids = gen_output_ids[:, input_ids.shape[1] :]
                new_text = self.generator.tokenizer.batch_decode(new_input_ids)
                new_scores, prm_state = self.prm(new_text, prm_state)

                if prm_scores is None:
                    prm_scores = new_scores[:, None]
                else:
                    prm_scores = torch.cat(
                        (
                            prm_scores.repeat(self.branching_factor, 1),
                            new_scores[:, None],
                        ),
                        dim=1,
                    )

                if self.score_aggregation == "min":
                    aggregate_scores, _ = torch.min(prm_scores, dim=1)
                elif self.score_aggregation == "mean":
                    aggregate_scores = torch.mean(prm_scores, dim=1)
                elif self.score_aggregation == "last":
                    aggregate_scores = prm_scores[:, -1]
                else:
                    raise NotImplementedError(
                        f"{self.score_aggregation} aggregation is not implemented."
                    )

                for i, aggregate_score in enumerate(aggregate_scores.tolist()):
                    if is_complete[i]:
                        child = ReasoningNode(
                            parent=node_selected,
                            children=[],
                            current_ids=gen_output_ids[[i]],
                            gen_state=None,
                            prm_state=None,
                            prm_scores=None,
                            score=aggregate_score,
                            is_leaf=True,
                        )
                        decoded_path = self.generator.decode(gen_output_ids[i])
                        step_scores = prm_scores[i].tolist()
                        complete_paths.append(
                            (decoded_path, aggregate_score, step_scores)
                        )
                    else:
                        child = ReasoningNode(
                            parent=node_selected,
                            children=[],
                            current_ids=gen_output_ids[[i]],
                            gen_state=self.generator.filter_state(gen_state, [i]),
                            prm_state=self.prm.filter_state(prm_state, [i]),
                            prm_scores=prm_scores[[i]],
                            score=aggregate_score,
                            is_leaf=False,
                        )

                    node_selected.children.append(child)

                node_selected = node_selected.children[torch.argmax(aggregate_scores)]
                depth += 1

            # backpropagation
            while node_selected.parent is not None:
                node_selected = node_selected.parent
                node_selected.n_visits += self.branching_factor
                node_selected.score = max(
                    child.score for child in node_selected.children
                )
                node_selected.free_state()

            if len(complete_paths) == 0:
                outputs.append(
                    {
                        "answer": None,
                        "outputs": [],
                        "aggregate_scores": [],
                        "step_scores": [],
                    }
                )
            else:
                complete_paths.sort(key=lambda t: t[1], reverse=True)
                outputs.append(
                    {
                        "answer": complete_paths[0][0],
                        "outputs": [t[0] for t in complete_paths],
                        "aggregate_scores": [t[1] for t in complete_paths],
                        "step_scores": [t[2] for t in complete_paths],
                    }
                )

        return outputs
