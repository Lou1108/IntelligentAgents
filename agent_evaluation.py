from sentence_transformers import SentenceTransformer, util
import pandas as pd
import json

bert_model = SentenceTransformer('all-MiniLM-L6-v2')


def safe_division(num, denom):
    return num / denom if denom != 0 else float('nan')


def compute_cosine_similarity(input_1: str, input_2: str) -> float:
    """
    computes the cosine similarity between two inputs using the BERT model
    :return: float in [0, 1]; 1 == identical meaning
    """

    # translate to embeddings
    emb_1 = bert_model.encode(input_1, convert_to_tensor=True)
    emb_2 = bert_model.encode(input_2, convert_to_tensor=True)
    return float(util.cos_sim(emb_1, emb_2).item())


def compute_metrics(story_number, IE, RE, NE, original_input, corrected_input):
    """
    computes a variety of metrics of a specific LLM correction
    :param story_number:
    :param IE: number of initial errors
    :param RE: number or remaining errors
    :param NE: numbre of new errors
    :param original_input: the original text input
    :param corrected_input: the LLM corrected story input
    :return: a dictionary of computed metrics
    """
    CE = IE - RE  # corrected errors
    correction_rate = safe_division(CE, IE)
    error_addition_rate = safe_division(NE, (IE + NE))

    precision = safe_division(CE, (CE + NE))
    recall = safe_division(CE, IE)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')

    semantic_score = compute_cosine_similarity(original_input,
                                               corrected_input) if original_input and corrected_input else float('nan')

    return {
        "story number": story_number,
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
    """
    retrieves the grouped and average metric results
    """
    individual_metrics = [compute_metrics(**ex) for ex in inputs if ex != "story_number"]
    df = pd.DataFrame(individual_metrics)

    # averages of each story for 10 runs
    grouped = df.groupby('story number').mean(numeric_only=True)

    return grouped, grouped.describe().loc[['mean', 'std']].T


def load_agent_input_from_file(filepath):
    """
    loads agent inputs from json file
    """
    with open(filepath, "r", encoding="utf-8") as f:
        stories = json.load(f)
    return stories


if __name__ == "__main__":
    # potentially requires: pip install "numpy<2.0" --force-reinstall

    print("------------------------- Evaluate Ontology Agent -------------------------")
    ontology_agent_results = load_agent_input_from_file("agent_evaluation/ontology_agent.json")
    grouped_ontology, avg_ontology = get_average_metrics(ontology_agent_results)

    print("------------------------- Evaluate Baseline Agent -------------------------")
    baseline_agent_results = load_agent_input_from_file("agent_evaluation/baselne_agent.json")
    grouped_baseline, avg_baseline = get_average_metrics(baseline_agent_results)

    print("------------------------- Generate Average Model Metrics -------------------------")
    comp_models = pd.concat([grouped_ontology.T, grouped_baseline.T], axis=1, ignore_index=True)
    comp_models.columns = ['Ontology (1)', 'Ontology (2)', 'Ontology (3)', 'Ontology (4)', 'Ontology (5)', 'Baseline (1)', 'Baseline (2)', 'Baseline (3)', 'Baseline (4)', 'Baseline (5)']
    comp_models.to_csv("agent_evaluation/model_comparison.csv")
    print(comp_models.to_string())

    # combined table for comparison
    print("------------------------- Generate Overall Model comparison -------------------------")
    comp_avg = pd.concat([avg_ontology, avg_baseline], axis=1, ignore_index=True)
    comp_avg.columns = ['Ontology Mean', 'Ontology Std', 'Baseline Mean', 'Baseline Std']
    comp_avg.to_csv("agent_evaluation/average_comparison.csv")
    print(comp_avg.to_string())
