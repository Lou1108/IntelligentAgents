# IntelligentAgents
This repository is used for a group project in the course Intelligent Agents at Utrecht University. In this project we will augment a basic language model with an agentic pipeline that builds upon an ontology.<br>

Reference Repository for connecting Ontology to Agent: https://github.com/sorindragan/ontology-tutorial<br>

---Files---<br>
agent.ipynb: Runs ontology driven agent<br>
base_agent.ipynb: Runs baseline agent (includes same steps as the ontology agent, except for reasoning with ontology using RAG)<br>
agent_evaluation.py: Creates evaluation metrics of both models<br>

---Folders---<br>
baseline_agent_output: Contains outputs of baseline model. Each story has been tested a total of 10 times, stored in files base_model_storyX_10runs.json, with X being the story (1-5).<br>
ontology_agent_output: Contains outputs of ontology model. Each story has been tested a total of 10 times, stored in files ontology_model_storyX_10runs.json, with X being the story (1-5).<br>

---Changing story received---<br>
To change the story received by the model, comment out all but 1 story.<br>

---Note---<br>
story+reasoning_output.json contains 1 example of the reasoning steps of the llm for fixing scenarios. This only includes the reasoning steps of the last loop of fixing scenarios.<br>
