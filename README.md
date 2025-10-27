# IntelligentAgents
This repository is used for a group project in the course Intelligent Agents at Utrecht University. In this project we will augment a basic language model with an agentic pipeline that builds upon an ontology.

Reference Repository for connecting Ontology to Agent: https://github.com/sorindragan/ontology-tutorial

---
## Project Structure

```
project-root/
├── agent_evaluation/            # Output fiels from agent evaluation
├── baseline_agent_output/       # Output files from Baseline agent
├── ontology_agent_output/       # Output files from ontology agent
│
├── agent.ipynb                  # Runs ontology driven agent
├── base_agent.ipynb             # Creates evaluation metrics of both models
├── agent_evaluation.py          # Runs baseline agent (includes same steps as the ontology agent, except for reasoning with ontology using RAG)
│
├── Ontology_Assignment.rdf      # RDF file for the entire ontology
├── story+reasoning_output.json  # contains 1 exmaple of resaoning steps from LLM output
│
└── README.md                    # This file: project documentation
```

### Folders

- baseline_agent_output: Contains outputs of baseline model. Each story has been tested a total of 10 times, stored in files base_model_storyX_10runs.json, with X being the story (1-5). 
- ontology_agent_output: Contains outputs of ontology model. Each story has been tested a total of 10 times, stored in files ontology_model_storyX_10runs.json, with X being the story (1-5).

### Note
story+reasoning_output.json contains 1 example of the reasoning steps of the llm for fixing scenarios. This only includes the reasoning steps of the last loop of fixing scenarios.

---
### Changing story received
To change the story received by the model, comment out all but 1 story.

---

## Code Execution
To run the project, execute the agent notebooks first, then run the evaluation script.

1. Run Notebooks
Use Jupyter Notebook to execute each event file:
```
jupyter notebook

agent.ipynb
base_agent.ipynb
```
2. Run Evaluation
The evaluation is based on human evaluation. For this, one has to manually analyze the agent story outputs. 
These results can then be stored in agent_evaluation/baseline_agent.json and agent_evaluation/ontology_agent.json respectively.
To generate the evaluation results, run the python script:

```
python agent_evaluation.py
```

This evaluates the results and stores them as .csv in the agent_evaluation/ directory.


