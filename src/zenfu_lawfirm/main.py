# Example usage in main.py:

from collaboration.agent_network import LegalAgentNetwork
from collaboration.workflow import create_workflow, AgentState

# Initialize the network
agent_network = LegalAgentNetwork(agents, llm)

# Create the workflow
workflow = create_workflow(agent_network)

# Execute the workflow
initial_state = AgentState(
    query="Review my employment contract with IP implications",
    primary_agent="contract_lawyer",
    consultations={},
    final_response=""
)

final_state = workflow.execute(initial_state)
print(final_state["final_response"])
