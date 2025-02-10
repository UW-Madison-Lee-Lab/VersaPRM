from typing import Any, Literal

import torch

from interfaces import BaseGenerator, BasePRM


class BeamSearch:
    """
    ### Args:
        generator (BaseGenerator): Generator LLM
        prm (BasePRM): PRM
        number_of_beams (int): Number of beams before filtering
        width (int): Number of beams sampled from each existing beam
        max_generation_rounds (int): Maximum number of generation rounds

    ### Algorithm:

    Beam search optimizes the PRM by searching over its per-step predictions.
    Concretely, we consider a fixed `number_of_beams` and a `beam_width`.
    We then run the following steps:

    1. sample `number_of_beams` initial predictions for the first step in the solution

    2. score the generated steps according to the PRM's predicted step-wise reward-to-go estimate

    3. filter for only the top `number_of_beams / beam_width` highest scoring steps

    4. now from each candidate, sample `beam_width` proposals from the next step, resulting in a total of `(number_of_beams / beam_width) * beam_width` candidate prefixes again

    5. repeat from step 2

    TODO: Cite algorithm
    """

    def __init__(
        self,
        generator: BaseGenerator,
        prm: BasePRM,
        number_of_beams: int = 8,
        width: int = 2,
        max_generation_rounds: int = 10,
        score_aggregation: Literal["min", "mean", "last"] = "min",
    ):
        assert number_of_beams % width == 0
        self.generator = generator
        self.prm = prm
        self.width = width  # `beam_width`
        self.number_of_beams = number_of_beams  # N
        self.max_generation_rounds = max_generation_rounds
        self.score_aggregation = score_aggregation

    def __call__(self, question: str) -> dict[str, Any]:
        input_ids = self.generator.encode(question)

        gen_state = self.generator.init_state(input_ids)
        prm_state = self.prm.init_state(question)

        input_ids = input_ids.repeat(self.number_of_beams, 1)
        gen_state = self.generator.inflate_state(gen_state, self.number_of_beams)
        prm_state = self.prm.inflate_state(prm_state, self.number_of_beams)
        scores = None

        complete_beams: list[tuple[str, float, list[float]]] = []

        for i in range(self.max_generation_rounds):
            gen_output_ids, gen_state = self.generator(input_ids, gen_state)
            is_complete = self.generator.is_complete(gen_output_ids)

            new_input_ids = gen_output_ids[:, input_ids.shape[1] :]
            new_text = self.generator.tokenizer.batch_decode(new_input_ids)
            new_scores, prm_state = self.prm(new_text, prm_state)

            if scores is None:
                scores = new_scores[:, None]
            else:
                scores = torch.cat((scores, new_scores[:, None]), dim=1)

            if self.score_aggregation == "min":
                aggregate_scores, _ = torch.min(scores, dim=1)
            elif self.score_aggregation == "mean":
                aggregate_scores = torch.mean(scores, dim=1)
            elif self.score_aggregation == "last":
                aggregate_scores = scores[:, -1]
            else:
                raise NotImplementedError(
                    f"{self.score_aggregation} aggregation is not implemented."
                )

            for i in range(len(is_complete)):
                if is_complete[i]:
                    decoded_beam = self.generator.decode(gen_output_ids[i])
                    aggregate_score = aggregate_scores[i].item()
                    step_scores = scores[i].tolist()
                    complete_beams.append((decoded_beam, aggregate_score, step_scores))

            if torch.all(is_complete):
                break
            else:
                incomplete_idxs = [i for i, c in enumerate(is_complete) if not c]
                gen_output_ids = gen_output_ids[incomplete_idxs]
                scores = scores[incomplete_idxs]
                gen_state = self.generator.filter_state(gen_state, incomplete_idxs)
                prm_state = self.prm.filter_state(prm_state, incomplete_idxs)

            sorted_scored_idxs = sorted(
                enumerate(scores.tolist()),
                key=lambda t: t[1],
                reverse=True,
            )
            sorted_idxs = [i for i, _ in sorted_scored_idxs]
            best_idxs = sorted_idxs[: self.number_of_beams // self.width]

            gen_output_ids = gen_output_ids[best_idxs]
            scores = scores[best_idxs]
            gen_state = self.generator.filter_state(gen_state, best_idxs)
            prm_state = self.prm.filter_state(prm_state, best_idxs)

            input_ids = gen_output_ids.repeat(self.width, 1)
            scores = scores.repeat(self.width, 1)
            gen_state = self.generator.inflate_state(gen_state, self.width)
            prm_state = self.prm.inflate_state(prm_state, self.width)

        if len(complete_beams) == 0:
            return {
                "answer": None,
                "outputs": [],
                "aggregate_scores": [],
                "step_scores": [],
            }
        else:
            complete_beams.sort(key=lambda t: t[1], reverse=True)
            return {
                "answer": complete_beams[0][0],
                "outputs": [t[0] for t in complete_beams],
                "aggregate_scores": [t[1] for t in complete_beams],
                "step_scores": [t[2] for t in complete_beams],
            }
