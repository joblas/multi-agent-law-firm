# src/zenfu_lawfirm/crew.py

import os
from crewai import Agent, Task, Crew, Process
from tools.document_tool import LegalDocSearchTool
from tools.web_search_tool import WebSearchTool
from tools.policy_monitor import PolicyMonitor
from ethics.bias_detector import BiasDetector
import yaml

class ZenFuLawFirm:
    def __init__(self):
        self.load_configurations()
        self.initialize_tools()
        self.initialize_agents()
        self.initialize_tasks()
        self.initialize_crew()

    def load_configurations(self):
        """Load configuration files"""
        config_path = os.path.join(os.path.dirname(__file__), 'config')
        
        with open(os.path.join(config_path, 'agents.yaml'), 'r') as f:
            self.agent_config = yaml.safe_load(f)
            
        with open(os.path.join(config_path, 'tasks.yaml'), 'r') as f:
            self.task_config = yaml.safe_load(f)
            
        with open(os.path.join(config_path, 'ethics.yaml'), 'r') as f:
            self.ethics_config = yaml.safe_load(f)

    def initialize_tools(self):
        """Initialize all tools"""
        self.doc_tool = LegalDocSearchTool('legal_docs')
        self.web_tool = WebSearchTool()
        self.policy_monitor = PolicyMonitor()
        self.bias_detector = BiasDetector()

    def initialize_agents(self):
        """Initialize all agents with their tools"""
        self.agents = []
        
        for agent_name, config in self.agent_config.items():
            tools = self._get_agent_tools(agent_name)
            agent = Agent(
                role=config['role'],
                goal=config['goal'],
                backstory=config['backstory'],
                tools=tools,
                verbose=True
            )
            self.agents.append(agent)

    def initialize_tasks(self):
        """Initialize all tasks"""
        self.tasks = []
        
        for task_name, config in self.task_config.items():
            agent = self._get_task_agent(task_name)
            task = Task(
                description=config['description'],
                expected_output=config['expected_output'],
                agent=agent
            )
            self.tasks.append(task)

    def initialize_crew(self):
        """Initialize the crew with all agents and tasks"""
        self.crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )

    def _get_agent_tools(self, agent_name: str) -> list:
        """Get appropriate tools for each agent type"""
        common_tools = [self.web_tool]
        
        if agent_name == 'manager':
            return common_tools + [self.policy_monitor]
        elif agent_name == 'civil_rights_lawyer':
            return common_tools + [self.doc_tool, self.policy_monitor]
        elif agent_name == 'ethics_reviewer':
            return common_tools + [self.bias_detector]
        
        return common_tools

    def _get_task_agent(self, task_name: str) -> Agent:
        """Get appropriate agent for each task type"""
        if task_name.startswith('handle_customer'):
            return self.agents[0]  # Manager
        elif task_name.startswith('civil_rights'):
            return self.agents[1]  # Civil Rights Lawyer
        elif task_name.startswith('ethics'):
            return self.agents[2]  # Ethics Reviewer
        
        return self.agents[0]  # Default to manager

    async def process_query(self, query: Dict) -> Dict:
        """Process a legal query through the system"""
        try:
            # Initial bias check
            bias_analysis = self.bias_detector.analyze_decision({
                'query': query['query_text'],
                'context': query
            })
            
            # Check if bias is within acceptable limits
            if bias_analysis['metrics']['bias_score'] < 0.3:
                # Process query
                response = await self.crew.kickoff(inputs=query)
                
                # Get policy updates
                policy_updates = await self.policy_monitor.monitor_policy_changes()
                
                return {
                    'status': 'success',
                    'response': response,
                    'bias_analysis': bias_analysis,
                    'policy_updates': policy_updates
                }
            else:
                return {
                    'status': 'error',
                    'error': 'Potential bias detected',
                    'bias_analysis': bias_analysis,
                    'recommendations': bias_analysis['recommendations']
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'details': 'An error occurred while processing the query'
            }
