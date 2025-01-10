import streamlit as st
import os
from datetime import datetime
from src.zenfu_lawfirm.collaboration.agent_network import LegalAgentNetwork
from src.zenfu_lawfirm.collaboration.workflow import create_workflow, AgentState
from src.zenfu_lawfirm.ethics.bias_detector import BiasDetector
from src.zenfu_lawfirm.ethics.transparency import TransparencyHandler
from src.zenfu_lawfirm.ethics.accountability import AccountabilityTracker
from src.zenfu_lawfirm.tools.policy_monitor import PolicyMonitor
from src.zenfu_lawfirm.tools.document_tool import DocumentTool
from src.zenfu_lawfirm.data_processing.data_curator import DataCurator

# Page configuration
st.set_page_config(
    page_title="ZenFu AI Lawfirm",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define tasks
TASKS = {
    "handle_customer_query": {
        "description": "Listen to the customer's question and identify the relevant legal area.",
        "expected_output": "A task delegation to the appropriate lawyer agent."
    },
    "civil_rights_review": {
        "description": "Review potential civil rights violations and provide comprehensive analysis.",
        "expected_output": "A detailed report on civil rights implications and recommended actions."
    },
    "ethics_review": {
        "description": "Monitor for potential biases and ensure ethical handling of the case.",
        "expected_output": "An ethics report including bias analysis and transparency recommendations."
    },
    "policy_monitoring": {
        "description": "Track relevant policy changes and assess their impact on current cases.",
        "expected_output": "A policy update report with impact analysis."
    }
}

# Initialize session state
if 'current_case' not in st.session_state:
    st.session_state.current_case = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'case_history' not in st.session_state:
    st.session_state.case_history = []
if 'current_task' not in st.session_state:
    st.session_state.current_task = None

# Initialize components
@st.cache_resource
def initialize_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        temperature=0.7,
        model="gpt-4-turbo-preview",
        streaming=True
    )

@st.cache_resource
def initialize_agents(llm):
    # Load agent configurations from YAML
    agents = {
        "manager": {
            "role": "Law Firm Manager",
            "expertise": ["task delegation", "client relations", "legal assessment"]
        },
        "civil_rights_lawyer": {
            "role": "Civil Rights Specialist",
            "expertise": ["civil rights law", "discrimination", "constitutional law", "policy analysis"]
        },
        "ethics_reviewer": {
            "role": "Ethics and Bias Monitor",
            "expertise": ["ethics review", "bias detection", "transparency", "accountability"]
        }
    }
    return LegalAgentNetwork(agents, llm)

@st.cache_resource
def load_components():
    llm = initialize_llm()
    return {
        'agent_network': initialize_agents(llm),
        'bias_detector': BiasDetector(),
        'transparency_handler': TransparencyHandler(),
        'accountability_tracker': AccountabilityTracker(),
        'policy_monitor': PolicyMonitor(),
        'document_tool': DocumentTool(),
        'data_curator': DataCurator()
    }

components = load_components()

# Main header
st.title("🏛️ ZenFu AI Lawfirm")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Select Page",
        ["Case Analysis", "Document Review", "Policy Monitor", "Ethics Dashboard"]
    )

# Initialize workflow
if 'workflow' not in st.session_state:
    st.session_state.workflow = create_workflow(components['agent_network'])

# Main content area based on selected page
if page == "Case Analysis":
    st.header("Case Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Case Documents", type=['pdf', 'docx', 'txt'])
    if uploaded_file:
        st.session_state.current_case = {
            'id': datetime.now().strftime("%Y%m%d%H%M%S"),
            'name': uploaded_file.name,
            'content': uploaded_file.read()
        }
        
    # Case input
    case_description = st.text_area("Case Description", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        case_type = st.selectbox(
            "Case Type",
            ["Civil Rights", "Contract Law", "Intellectual Property", "Criminal Law", "Data Privacy"]
        )
    with col2:
        priority = st.select_slider(
            "Priority Level",
            options=["Low", "Medium", "High", "Urgent"]
        )
    
    # Case routing
    st.subheader("Case Assignment")
    case_category = st.selectbox(
        "Case Category",
        [
            "Civil Rights",
            "Constitutional Law",
            "Discrimination",
            "Policy Analysis",
            "Other"
        ]
    )
    
    # Automatically select primary agent based on case category
    primary_agent = "civil_rights_lawyer"  # Default for most cases
    if case_category == "Other":
        primary_agent = "manager"  # Manager handles general inquiries
        
    st.info(f"Your case will be handled by our {components['agent_network'].agents[primary_agent]['role']}")
    
    # Ethics review option
    include_ethics_review = st.checkbox(
        "Include Ethics Review",
        value=True,
        help="Have our Ethics and Bias Monitor review the case"
    )
    
    if st.button("Analyze Case"):
        with st.spinner("Analyzing case with legal experts..."):
            # Process the case with agent network
            if st.session_state.current_case and case_description:
                # Initialize agent state
                initial_state = AgentState(
                    query=case_description,
                    primary_agent=primary_agent,
                    consultations={},
                    final_response=""
                )
                
                if include_ethics_review:
                    initial_state["ethics_review"] = True
                
                # Run the workflow
                final_state = st.session_state.workflow.run(initial_state)
                st.session_state.analysis_results = final_state
                
                # Display results in organized tabs
                tab1, tab2, tab3 = st.tabs(["Primary Analysis", "Ethics Review", "Final Opinion"])
                
                with tab1:
                    st.subheader(f"Analysis by {components['agent_network'].agents[primary_agent]['role']}")
                    st.write(final_state["consultations"][primary_agent])
                
                with tab2:
                    if include_ethics_review and "ethics_reviewer" in final_state["consultations"]:
                        st.subheader("Ethics Review")
                        st.write(final_state["consultations"]["ethics_reviewer"])
                        
                        # Display any bias warnings or ethical considerations
                        if components['bias_detector'].has_bias_concerns(final_state["consultations"][primary_agent]):
                            st.warning("⚠️ Potential bias detected. See ethics review for details.")
                
                with tab3:
                    st.subheader("Synthesized Legal Opinion")
                    st.write(final_state["final_response"])
                    
                    # Add accountability tracking
                    if st.session_state.current_case:
                        components['accountability_tracker'].log_decision(
                            st.session_state.current_case['id'],
                            {
                                "primary_agent": primary_agent,
                                "ethics_review": include_ethics_review,
                                "case_category": case_category,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                
                # Bias check
                bias_report = components['bias_detector'].analyze_decision(analysis)
                
                # Display results
                st.subheader("Analysis Results")
                st.json(analysis)
                
                # Show bias analysis
                st.subheader("Bias Analysis")
                st.write(bias_report)

elif page == "Document Review":
    st.header("Document Review")
    
    # Document management interface
    uploaded_files = st.file_uploader(
        "Upload Legal Documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for doc in uploaded_files:
            st.write(f"Processing: {doc.name}")
            doc_analysis = components['document_tool'].analyze_document(doc.read())
            st.json(doc_analysis)

elif page == "Policy Monitor":
    st.header("Policy Monitor")
    
    # Policy monitoring interface
    if st.button("Check Recent Policy Changes"):
        with st.spinner("Monitoring policy changes..."):
            updates = components['policy_monitor'].monitor_policy_changes()
            
            st.subheader("Recent Policy Updates")
            for update in updates:
                st.write(f"- {update}")
    
    # Policy alert settings
    st.subheader("Alert Settings")
    st.multiselect(
        "Select areas for policy monitoring",
        ["Civil Rights", "Data Privacy", "Criminal Law", "Corporate Law"]
    )

elif page == "Ethics Dashboard":
    st.header("Ethics Dashboard")
    
    # Ethics monitoring tabs
    tab1, tab2, tab3 = st.tabs(["Bias Analysis", "Transparency", "Accountability"])
    
    with tab1:
        st.subheader("Bias Detection Results")
        if st.session_state.analysis_results:
            bias_report = components['bias_detector'].analyze_decision(
                st.session_state.analysis_results
            )
            st.write(bias_report)
    
    with tab2:
        st.subheader("Decision Transparency")
        if st.session_state.current_case:
            explanation = components['transparency_handler'].explain_decision(
                st.session_state.current_case,
                st.session_state.analysis_results
            )
            st.write(explanation)
    
    with tab3:
        st.subheader("Accountability Tracking")
        if st.session_state.current_case:
            report = components['accountability_tracker'].generate_accountability_report(
                st.session_state.current_case['id']
            )
            st.write(report)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>ZenFu AI Lawfirm - Powered by AI for Justice</p>
    </div>
    """,
    unsafe_allow_html=True
)