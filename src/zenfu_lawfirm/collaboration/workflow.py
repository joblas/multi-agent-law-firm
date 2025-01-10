# src/zenfu_lawfirm/collaboration/workflow.py
from langgraph.graph import Graph
from langgraph.prebuilt import ToolExecutor
from typing import Dict, TypedDict

class AgentState(TypedDict):
    """State of the agent workflow"""
    query: str
    primary_agent: str
    consultations: Dict[str, str]
    final_response: str

def create_workflow(agent_network):
    """Create a workflow for agent collaboration"""
    # Define workflow nodes
    workflow = Graph()

    # Add nodes for each step in the process
    @workflow.node
    def initial_analysis(state: AgentState) -> AgentState:
        """Primary agent's initial analysis"""
        response = agent_network.agent_executors[state["primary_agent"]].run(
            state["query"]
        )
        state["consultations"][state["primary_agent"]] = response
        return state

    @workflow.node
    def consult_experts(state: AgentState) -> AgentState:
        """Consult other relevant experts"""
        query = state["query"]
        primary = state["primary_agent"]
        
        for agent_name, executor in agent_network.agent_executors.items():
            if agent_name != primary and agent_network._should_consult_agent(
                query, primary, agent_name
            ):
                consultation = executor.run(query)
                state["consultations"][agent_name] = consultation
        
        return state

    @workflow.node
    def synthesize_response(state: AgentState) -> AgentState:
        """Synthesize final response from all consultations"""
        state["final_response"] = agent_network._synthesize_response(
            state["primary_agent"],
            state["consultations"][state["primary_agent"]],
            [v for k, v in state["consultations"].items() if k != state["primary_agent"]]
        )
        return state

    # Define workflow edges
    workflow.add_edge("initial_analysis", "consult_experts")
    workflow.add_edge("consult_experts", "synthesize_response")

    return workflow

