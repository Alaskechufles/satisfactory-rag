"""Simulated retriever for tests and for the US0.2.3 demo."""


class FixedRetriever:
    """Returns fixed ids per question and records what it was asked."""

    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, k: int) -> list[str]:
        self.calls.append((question, k))
        return self.answers.get(question, [])


# Against fixtures/sample_dataset.jsonl, worked out by hand with k=5:
#   Q001 finds 1 of 2, at rank 2  -> recall 1/2, RR 1/2
#   Q002 finds 1 of 1, at rank 1  -> recall 1,   RR 1
#   Q003 finds 1 of 1, at rank 3  -> recall 1,   RR 1/3
# Conceptual + lookup: recall (1/2 + 1) / 2 = 0.750, MRR (1/2 + 1) / 2 = 0.750
# Guide:               recall 1.000,                 MRR 0.333
SAMPLE_ANSWERS = {
    "¿Qué diferencia hay entre el Constructor y la Ensambladora?": [
        "Desc_Rotor_C",
        "Build_ConstructorMk1_C",
        "Desc_IronRod_C",
        "Desc_IronPlate_C",
        "Desc_Wire_C",
        "Build_AssemblerMk1_C",  # rank 6: outside the top 5, must not count
    ],
    "¿Qué necesito para fabricar Hormigón?": ["Recipe_Concrete_C", "Desc_Cement_C"],
    "¿Qué mejora del HUB desbloquea la Cinta transportadora Mk.1?": [
        "Schematic_Tutorial1_C",
        "Schematic_Tutorial2_C",
        "Schematic_Tutorial3_C",
    ],
}

sample_retriever = FixedRetriever(SAMPLE_ANSWERS)
