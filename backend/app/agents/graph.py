from langgraph.graph import StateGraph, END
from app.agents.state import ClaimState
from app.agents.nodes import (
    intent_agent, extraction_agent, coding_agent, 
    validation_agent, prediction_agent, correction_agent, explanation_agent
)

def build_graph():
    """
    Compiles the directed graph detailing how the text flows through the agents.
    """
    workflow = StateGraph(ClaimState)

    # 1. Add Nodes
    workflow.add_node("intent", intent_agent)
    workflow.add_node("extraction", extraction_agent)
    workflow.add_node("coding", coding_agent)
    workflow.add_node("validation", validation_agent)
    workflow.add_node("prediction", prediction_agent)
    workflow.add_node("correction", correction_agent)
    workflow.add_node("explanation", explanation_agent)

    # 2. Add Routing and Edges
    # Start at intent
    workflow.set_entry_point("intent")
    
    # After intent -> extraction (assuming it's a NEW_CLAIM for now)
    workflow.add_edge("intent", "extraction")
    
    # After extraction -> coding
    workflow.add_edge("extraction", "coding")
    
    # After coding -> validation
    workflow.add_edge("coding", "validation")
    
    # After validation -> prediction
    workflow.add_edge("validation", "prediction")
    
    # After prediction -> correction
    workflow.add_edge("prediction", "correction")
    
    # After correction -> explanation
    workflow.add_edge("correction", "explanation")
    
    # After explanation -> END
    workflow.add_edge("explanation", END)

    # Compile the orchestrator
    orchestrator = workflow.compile()
    return orchestrator

# Expose a ready-to-run instance
agent_orchestrator = build_graph()
