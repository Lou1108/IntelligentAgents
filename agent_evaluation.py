from sentence_transformers import SentenceTransformer, util
import pandas as pd
import json

bert_model = SentenceTransformer('all-MiniLM-L6-v2')


def safe_division(num, denom):
    return num / denom if denom != 0 else float('nan')


def compute_cosine_similarity(input_1: str, input_2: str) -> float:
    """
    computes the cosine similarity between two inputs using the BERT model
    :param input_1:
    :param input_2:
    :return: float in [0, 1]; 1 == identical meaning
    """

    # translate to embeddings
    emb_1 = bert_model.encode(input_1, convert_to_tensor=True)
    emb_2 = bert_model.encode(input_2, convert_to_tensor=True)
    return float(util.cos_sim(emb_1, emb_2).item())


def compute_metrics(IE, RE, NE, original_input, corrected_input):
    CE = IE - RE  # corrected errors
    correction_rate = safe_division(CE, IE)
    error_addition_rate = safe_division(NE, (IE + NE))

    precision = safe_division(CE, (CE + NE))
    recall = safe_division(CE, IE)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')

    semantic_score = compute_cosine_similarity(original_input,
                                               corrected_input) if original_input and corrected_input else float('nan')

    return {
        "E (Initial Errors)": IE,
        "RE (Remaining Errors)": RE,
        "NE (New Errors)": NE,
        "CE (Corrected Errors)": CE,
        "Correction Rate": correction_rate,
        "Error Addition Rate": error_addition_rate,
        "Precision": precision,
        "Recall": recall,
        "F1": f1_score,
        "Semantic Preservation (SPS)": semantic_score
    }


def get_average_metrics(inputs):
    individual_metrics = [compute_metrics(**ex) for ex in inputs]
    df = pd.DataFrame(individual_metrics)

    return df.describe().loc[['mean', 'std']].T


def load_agent_input_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        stories = json.load(f)
    return stories


if __name__ == "__main__":
    # potentially requires: pip install "numpy<2.0" --force-reinstall

    print("------------------------- Evaluate Ontology Agent -------------------------")
    ontology_agent_results = load_agent_input_from_file("agent_outputs/ontology_agent.json")
    print(get_average_metrics(ontology_agent_results))

    print("------------------------- Evaluate Baseline Agent -------------------------")
    baseline_agent_results = load_agent_input_from_file("agent_outputs/baselne_agent.json")
    print(get_average_metrics(baseline_agent_results))
