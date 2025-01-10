import streamlit as st
import os
from datetime import datetime
from src.zenfu_lawfirm.collaboration.agent_network import LegalAgentNetwork
from src.zenfu_lawfirm.collaboration.workflow import create_workflow, AgentState
from src.zenfu_lawfirm.ethics.bias_detector import BiasDetector
from src.zenfu_lawfirm.tools.policy_monitor import PolicyMonitor
from src.zenfu_lawfirm.data_processing.data_curator import DataCurator
from src.zenfu_lawfirm.tools.legal_tool_manager import LegalToolManager
from src.zenfu_lawfirm.tools.document_tool import LegalDocSearchTool

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
        'legal_tool_manager': LegalToolManager(docs_path='legal_docs'),
        'policy_monitor': PolicyMonitor(),
        'document_tool': LegalDocSearchTool(docs_path='legal_docs'),
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
            try:
                # Process the case with agent network
                if st.session_state.current_case and case_description:
                    # Perform legal analysis
                    legal_analysis = components['legal_tool_manager'].legal_analyzer.analyze_case(case_description)
                    
                    # Initialize agent state
                    initial_state = AgentState(
                        query=case_description,
                        primary_agent=primary_agent,
                        consultations={},
                        final_response=""
                    )
                    
                    if include_ethics_review:
                        initial_state["ethics_review"] = True
                        # Perform bias detection
                        bias_analysis = components['bias_detector'].detect_bias(case_description)
                    
                    # Run the workflow
                    final_state = st.session_state.workflow.run(initial_state)
                    st.session_state.analysis_results = final_state
                    
                    # Display results in organized tabs
                    tab1, tab2, tab3 = st.tabs(["Primary Analysis", "Ethics Review", "Final Opinion"])
                    
                    with tab1:
                        st.subheader(f"Analysis by {components['agent_network'].agents[primary_agent]['role']}")
                        st.write(legal_analysis['analysis'])
                        st.write(final_state["consultations"][primary_agent])
                    
                    with tab2:
                        if include_ethics_review:
                            st.subheader("Ethics Review")
                            st.write(final_state["consultations"].get("ethics_reviewer", "No ethics review available"))
                            
                            # Display bias analysis results
                            if bias_analysis.get('bias_detected'):
                                st.warning("⚠️ Potential bias detected. See details below.")
                                st.write(bias_analysis['recommendations'])
                    
                    with tab3:
                        st.subheader("Synthesized Legal Opinion")
                        st.write(final_state["final_response"])
                        
                        # Add recommendations
                        st.subheader("Recommendations")
                        st.write(legal_analysis['recommendations'])
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")

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
            try:
                # Get file type from name
                file_type = doc.name.split('.')[-1].lower()
                
                # Read document content
                content = doc.read()
                
                # Analyze document
                doc_analysis = components['document_tool'].analyze_document(content)
                
                if doc_analysis['status'] == 'success':
                    st.success(f"Successfully processed {doc.name}")
                    
                    # Display analysis results in an organized way
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Word Count", doc_analysis['word_count'])
                        st.text(f"File Type: {file_type}")
                    with col2:
                        st.text("Date Analyzed:")
                        st.text(doc_analysis['date_analyzed'])
                    
                    # Display document summary in expandable section
                    with st.expander("Document Summary"):
                        st.write(doc_analysis['summary'])
                else:
                    st.error(f"Error analyzing {doc.name}: {doc_analysis['error']}")
                    
            except Exception as e:
                st.error(f"Error processing document {doc.name}: {str(e)}")

elif page == "Policy Monitor":
    st.header("Policy Monitor")
    
    # Policy monitoring interface
    if st.button("Check Recent Policy Changes"):
        with st.spinner("Monitoring policy changes..."):
            try:
                updates = components['policy_monitor'].monitor_policy_changes()
                
                st.subheader("Recent Policy Updates")
                for update in updates:
                    st.write(f"- {update}")
            except Exception as e:
                st.error(f"Error checking policy changes: {str(e)}")
    
    # Policy alert settings
    st.subheader("Alert Settings")
    st.multiselect(
        "Select areas for policy monitoring",
        ["Civil Rights", "Data Privacy", "Criminal Law", "Corporate Law"]
    )

elif page == "Ethics Dashboard":
    st.header("Ethics Dashboard")
    
    # Ethics monitoring tabs
    tab1, tab2 = st.tabs(["Bias Analysis", "Case History"])
    
    with tab1:
        st.subheader("Bias Detection Results")
        if st.session_state.analysis_results:
            try:
                bias_analysis = components['bias_detector'].detect_bias(
                    st.session_state.analysis_results["final_response"]
                )
                st.write(bias_analysis)
            except Exception as e:
                st.error(f"Error performing bias analysis: {str(e)}")
    
    with tab2:
        st.subheader("Recent Case History")
        for case in st.session_state.case_history:
            st.write(f"Case ID: {case['id']}")
            st.write(f"Status: {case.get('status', 'Unknown')}")
            st.write("---")

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
