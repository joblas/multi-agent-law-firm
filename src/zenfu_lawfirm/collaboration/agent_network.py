# src/zenfu_lawfirm/collaboration/agent_network.py
from typing import Dict, List
from langchain.agents import AgentExecutor, initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import LLMChain
from langchain.graphs import NetworkxEntityGraph
from langchain.schema import SystemMessage, HumanMessage

class LegalAgentNetwork:
    def __init__(self, agents: Dict, llm):
        self.agents = agents
        self.llm = llm
        self.graph = self._build_agent_graph()
        self.agent_executors = self._initialize_agent_executors()

    def _build_agent_graph(self):
        """Build a graph representing agent relationships and expertise areas"""
        graph = NetworkxEntityGraph()
        
        # Add agents as nodes
        for agent_name, agent in self.agents.items():
            graph.add_node(agent_name, type="agent", expertise=agent.expertise)
        
        # Add relationships between agents
        relationships = {
            "contract_lawyer": ["ip_lawyer", "criminal_lawyer"],  # Contract lawyer can consult IP and Criminal
            "ip_lawyer": ["contract_lawyer"],  # IP lawyer often needs contract expertise
            "criminal_lawyer": ["contract_lawyer", "ip_lawyer"]  # Criminal might need both
        }
        
        for agent, connections in relationships.items():
            for connected_agent in connections:
                graph.add_edge(agent, connected_agent, relationship="can_consult")
        
        return graph

    def _initialize_agent_executors(self):
        """Initialize agent executors with tools for collaboration"""
        executors = {}
        for agent_name, agent in self.agents.items():
            tools = self._create_collaboration_tools(agent_name)
            executor = AgentExecutor.from_agent_and_tools(
                agent=initialize_agent(
                    tools,
                    self.llm,
                    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True
                ),
                tools=tools,
                verbose=True
            )
            executors[agent_name] = executor
        return executors

    def _create_collaboration_tools(self, agent_name: str) -> List[Tool]:
        """Create tools for inter-agent collaboration"""
        tools = []
        
        # Add tools to consult other agents
        for other_agent in self.agents:
            if other_agent != agent_name:
                tools.append(
                    Tool(
                        name=f"consult_{other_agent}",
                        func=lambda q, agent=other_agent: self.consult_agent(agent, q),
                        description=f"Consult the {other_agent} for their expertise on a specific matter"
                    )
                )
        
        # Add tool for peer review
        tools.append(
            Tool(
                name="request_peer_review",
                func=lambda x: self.request_peer_review(agent_name, x),
                description="Request a peer review of your analysis from other lawyers"
            )
        )
        
        return tools

    async def collaborative_response(self, query: str, primary_agent: str):
        """Generate a response with potential collaboration from other agents"""
        executor = self.agent_executors[primary_agent]
        
        # Initial analysis by primary agent
        initial_response = await executor.arun(
            input=f"Analyze this legal query: {query}. Consider if you need input from other specialists."
        )
        
        # Check if other agents should be consulted
        consultations = []
        connected_agents = self.graph.get_connected_nodes(primary_agent)
        
        for connected_agent in connected_agents:
            if self._should_consult_agent(query, primary_agent, connected_agent):
                consultation = await self.consult_agent(connected_agent, query)
                consultations.append(consultation)
        
        # Synthesize final response
        final_response = await self._synthesize_response(
            primary_agent,
            initial_response,
            consultations
        )
        
        return final_response

    def _should_consult_agent(self, query: str, primary_agent: str, other_agent: str) -> bool:
        """Determine if another agent should be consulted based on the query"""
        # Implementation would check query against agent expertise areas
        expertise_keywords = {
            "contract_lawyer": ["contract", "agreement", "terms"],
            "ip_lawyer": ["patent", "copyright", "trademark", "intellectual property"],
            "criminal_lawyer": ["criminal", "privacy", "rights", "trespassing"]
        }
        
        other_expertise = expertise_keywords.get(other_agent, [])
        return any(keyword in query.lower() for keyword in other_expertise)

    async def consult_agent(self, agent_name: str, query: str) -> str:
        """Consult another agent for their expertise"""
        executor = self.agent_executors[agent_name]
        return await executor.arun(
            input=f"Provide your expertise as {agent_name} on: {query}"
        )

    async def request_peer_review(self, agent_name: str, analysis: str) -> str:
        """Request peer review from other agents"""
        reviews = []
        for reviewer_name, executor in self.agent_executors.items():
            if reviewer_name != agent_name:
                review = await executor.arun(
                    input=f"Review this legal analysis as {reviewer_name}: {analysis}"
                )
                reviews.append(f"{reviewer_name}: {review}")
        return "\n\n".join(reviews)

    async def _synthesize_response(
        self,
        primary_agent: str,
        initial_response: str,
        consultations: List[str]
    ) -> str:
        """Synthesize a final response incorporating all inputs"""
        synthesis_prompt = (
            f"Primary analysis from {primary_agent}:\n{initial_response}\n\n"
            "Additional expert consultations:\n" +
            "\n".join(consultations) +
            "\n\nSynthesize a comprehensive response incorporating all perspectives."
        )
        
        chain = LLMChain(llm=self.llm, prompt=synthesis_prompt)
        return await chain.arun(input=synthesis_prompt)
