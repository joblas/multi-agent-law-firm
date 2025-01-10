# 🏛️ ZenFu AI Lawfirm

A sophisticated AI-powered law firm leveraging CrewAI, LangChain, and advanced machine learning to provide comprehensive legal services with a focus on civil rights and ethical considerations.

## ✨ Key Features

🤝 **Multi-Agent Collaboration**
- Manager agent for task delegation
- Specialized legal experts
- Ethics reviewer for bias monitoring
- Inter-agent consultation system

👨‍⚖️ **Legal Expertise Areas**
- Civil Rights Law
- Contract Law
- Intellectual Property
- Criminal Law
- Data Privacy & Protection

📚 **Advanced Capabilities**
- RAG (Retrieval Augmented Generation)
- Real-time policy monitoring
- Bias detection and mitigation
- Document analysis and processing
- Session persistence with unique IDs

🔍 **Ethical Framework**
- Bias detection and mitigation
- Transparency in decision-making
- Accountability tracking
- Privacy protection
- Fair representation analysis

## 📁 Project Structure
```
zenfu_ai_lawfirm/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── session_data/          # Session information
├── src/
│   ├── zenfu_lawfirm/
│   │   ├── config/
│   │   │   ├── agents.yaml
│   │   │   ├── tasks.yaml
│   │   │   └── ethics.yaml
│   │   ├── collaboration/
│   │   │   ├── __init__.py
│   │   │   ├── agent_network.py
│   │   │   └── workflow.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── document_tool.py
│   │   │   ├── web_search_tool.py
│   │   │   ├── policy_monitor.py
│   │   │   └── bias_checker.py
│   │   ├── data_processing/
│   │   │   ├── __init__.py
│   │   │   ├── data_curator.py
│   │   │   ├── bias_mitigation.py
│   │   │   └── privacy_handler.py
│   │   ├── ethics/
│   │   │   ├── __init__.py
│   │   │   ├── bias_detector.py
│   │   │   ├── transparency.py
│   │   │   └── accountability.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── model_trainer.py
│   │   │   └── data_validator.py
│   │   ├── __init__.py
│   │   ├── crew.py
│   │   └── main.py
├── legal_docs/            # Legal document storage
├── training_data/         # Training data
│   ├── civil_rights_cases/
│   ├── legal_precedents/
│   └── policy_documents/
├── models/               # Model storage
│   ├── bias_detection/
│   └── legal_analysis/
└── streamlit_app.py     # User interface
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- SerperDevTool API key
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/zenfu-ai-lawfirm.git
cd zenfu-ai-lawfirm
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Initialize the system:
```bash
python -m src.zenfu_lawfirm.training.data_validator
```

6. Run the application:
```bash
streamlit run streamlit_app.py
```

### Development in GitHub Codespaces

1. Open the repository in Codespaces
2. Environment variables will be loaded from Codespace secrets
3. Install dependencies and run as above
4. Access via the forwarded port (typically 8501)

## 🎯 Core Functionalities

### Civil Rights Focus
- Case analysis and violation detection
- Policy monitoring and impact assessment
- Rights protection strategies
- Educational resources

### Ethical Framework
- Bias detection and mitigation
- Transparency in decision-making
- Accountability tracking
- Privacy protection

### Data Processing
- Advanced data curation
- Bias mitigation in training data
- Privacy-preserving processing
- Continuous learning

### Multi-Agent Collaboration
- Expertise-based routing
- Peer review system
- Cross-domain consultation
- Collective decision-making

## 📊 Features and Capabilities

### 1. Legal Analysis
- Document review and analysis
- Case law research
- Policy impact assessment
- Rights violation detection

### 2. Policy Monitoring
- Real-time policy tracking
- Legislative updates
- Impact analysis
- Automated alerts

### 3. Ethical Safeguards
- Bias detection
- Fairness metrics
- Transparency reports
- Accountability tracking

### 4. Data Privacy
- Data anonymization
- Secure processing
- Access controls
- Audit trails

## 🛠️ Technical Components

### RAG Integration
- Document indexing
- Context retrieval
- Response generation
- Citation support

### Agent System
- Task delegation
- Expertise routing
- Collaborative problem-solving
- Knowledge sharing

### User Interface
- Interactive query system
- Document upload
- Progress tracking
- Result visualization

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For assistance:
1. Open an issue in the repository
2. Contact our support team
3. Check the documentation

## 🎉 Acknowledgments

- Built with CrewAI
- Enhanced with LangChain
- Powered by OpenAI
- UI by Streamlit
