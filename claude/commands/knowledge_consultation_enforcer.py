#!/usr/bin/env python3
"""
Knowledge Consultation Enforcement System for Issue #113
Critical missing piece: Blocking mechanism to ensure agents actually use knowledge database
"""

import json
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set


class KnowledgeConsultationEnforcer:
    """
    Enforces mandatory knowledge consultation for all RIF agents.
    Prevents agents from making decisions without consulting knowledge database.
    """
    
    def __init__(self, enforcement_mode="strict"):
        self.enforcement_mode = enforcement_mode  # "strict", "warning", "monitoring"
        self.session_id = f"session_{int(time.time())}"
        self.consultation_log = "/Users/cal/DEV/RIF/knowledge/consultation_log.json"
        self.agent_sessions = {}
        
        # Required knowledge consultation tools
        self.required_mcp_tools = {
            "mcp__rif-knowledge__get_claude_documentation",
            "mcp__rif-knowledge__query_knowledge", 
            "mcp__rif-knowledge__check_compatibility"
        }
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for knowledge consultation enforcement."""
        log_dir = Path("/Users/cal/DEV/RIF/knowledge/enforcement_logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"enforcement_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("KnowledgeEnforcer")
    
    def start_agent_session(self, agent_name: str, issue_id: str, task_description: str):
        """
        Start a new agent session with knowledge consultation requirements.
        This must be called at the beginning of any agent work.
        """
        session_data = {
            "agent_name": agent_name,
            "issue_id": issue_id,
            "task_description": task_description,
            "session_start": datetime.now().isoformat(),
            "knowledge_consulted": False,
            "mcp_tools_used": set(),
            "consultation_evidence": {},
            "decisions_made": 0,
            "blocked_decisions": 0,
            "status": "active"
        }
        
        session_key = f"{agent_name}_{issue_id}_{self.session_id}"
        self.agent_sessions[session_key] = session_data
        
        self.logger.info(f"Started session for {agent_name} on issue #{issue_id}")
        return session_key
    
    def record_knowledge_consultation(self, session_key: str, mcp_tool: str, query: str, result: Dict):
        """
        Record when an agent uses knowledge consultation tools.
        This proves the agent is following protocols.
        """
        if session_key not in self.agent_sessions:
            raise ValueError(f"Invalid session key: {session_key}")
        
        session = self.agent_sessions[session_key]
        session["mcp_tools_used"].add(mcp_tool)
        session["consultation_evidence"][mcp_tool] = {
            "query": query,
            "result_summary": str(result)[:200],  # Truncated for logging
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if minimum consultation requirements are met
        if len(session["mcp_tools_used"]) >= 2:  # At least 2 different tools used
            session["knowledge_consulted"] = True
            session["status"] = "consultation_complete"
        
        self.logger.info(f"Knowledge consultation recorded: {mcp_tool} by {session['agent_name']}")
        self._save_consultation_log()
    
    def request_decision_approval(self, session_key: str, decision_type: str, decision_details: str) -> bool:
        """
        Agent requests approval to make a decision.
        Returns True if allowed, False if blocked due to insufficient knowledge consultation.
        """
        if session_key not in self.agent_sessions:
            raise ValueError(f"Invalid session key: {session_key}")
        
        session = self.agent_sessions[session_key]
        session["decisions_made"] += 1
        
        # Check if knowledge consultation requirements are met
        consultation_complete = session["knowledge_consulted"]
        min_tools_used = len(session["mcp_tools_used"]) >= 1  # At least one tool
        
        if consultation_complete and min_tools_used:
            # Decision approved
            self.logger.info(f"Decision APPROVED for {session['agent_name']}: {decision_type}")
            return True
        else:
            # Decision blocked
            session["blocked_decisions"] += 1
            session["status"] = "blocked_insufficient_consultation"
            
            missing_requirements = []
            if not min_tools_used:
                missing_requirements.append(f"No MCP knowledge tools used (need at least 1)")
            if not consultation_complete:
                missing_requirements.append(f"Insufficient consultation depth")
            
            block_message = f"""
🚫 DECISION BLOCKED: Knowledge Consultation Required

Agent: {session['agent_name']}
Issue: #{session['issue_id']}
Decision: {decision_type}
Reason: {', '.join(missing_requirements)}

Required Actions:
1. Use mcp__rif-knowledge__get_claude_documentation for Claude Code capabilities
2. Use mcp__rif-knowledge__query_knowledge for patterns and solutions  
3. Use mcp__rif-knowledge__check_compatibility for approach validation
4. Provide evidence template showing consultation results

Until these requirements are met, no decisions or implementations are allowed.
            """
            
            self.logger.warning(f"Decision BLOCKED for {session['agent_name']}: {decision_type}")
            print(block_message)
            
            if self.enforcement_mode == "strict":
                return False
            else:
                # In warning mode, allow but log
                return True
    
    def generate_consultation_report(self, session_key: str) -> Dict:
        """Generate a report of knowledge consultation activities for an agent session."""
        if session_key not in self.agent_sessions:
            return {"error": "Invalid session key"}
        
        session = self.agent_sessions[session_key]
        
        report = {
            "session_summary": {
                "agent": session["agent_name"],
                "issue": session["issue_id"], 
                "task": session["task_description"],
                "duration": self._calculate_session_duration(session),
                "status": session["status"]
            },
            "knowledge_consultation": {
                "consultation_complete": session["knowledge_consulted"],
                "mcp_tools_used": list(session["mcp_tools_used"]),
                "required_tools_coverage": len(session["mcp_tools_used"]) / len(self.required_mcp_tools),
                "evidence_provided": len(session["consultation_evidence"])
            },
            "decision_tracking": {
                "decisions_attempted": session["decisions_made"],
                "decisions_blocked": session["blocked_decisions"],
                "decision_approval_rate": self._calculate_approval_rate(session)
            },
            "compliance_score": self._calculate_compliance_score(session),
            "recommendations": self._generate_recommendations(session)
        }
        
        return report
    
    def _calculate_session_duration(self, session: Dict) -> str:
        """Calculate how long the agent session has been active."""
        start_time = datetime.fromisoformat(session["session_start"])
        duration = datetime.now() - start_time
        return str(duration).split('.')[0]  # Remove microseconds
    
    def _calculate_approval_rate(self, session: Dict) -> float:
        """Calculate the percentage of decisions that were approved."""
        if session["decisions_made"] == 0:
            return 100.0
        approved = session["decisions_made"] - session["blocked_decisions"]
        return (approved / session["decisions_made"]) * 100.0
    
    def _calculate_compliance_score(self, session: Dict) -> float:
        """Calculate overall compliance score for knowledge consultation."""
        score = 0.0
        
        # Knowledge consultation completeness (50 points)
        if session["knowledge_consulted"]:
            score += 50.0
        else:
            score += (len(session["mcp_tools_used"]) / len(self.required_mcp_tools)) * 50.0
        
        # Evidence quality (25 points)  
        if len(session["consultation_evidence"]) > 0:
            score += 25.0
        
        # Decision approval rate (25 points)
        score += (self._calculate_approval_rate(session) / 100.0) * 25.0
        
        return min(score, 100.0)
    
    def _generate_recommendations(self, session: Dict) -> List[str]:
        """Generate recommendations to improve knowledge consultation compliance."""
        recommendations = []
        
        if not session["knowledge_consulted"]:
            recommendations.append("Complete knowledge consultation using all required MCP tools")
        
        if "mcp__rif-knowledge__get_claude_documentation" not in session["mcp_tools_used"]:
            recommendations.append("Query Claude Code documentation for capabilities understanding")
        
        if "mcp__rif-knowledge__query_knowledge" not in session["mcp_tools_used"]:
            recommendations.append("Search knowledge database for relevant patterns and solutions")
        
        if "mcp__rif-knowledge__check_compatibility" not in session["mcp_tools_used"]:
            recommendations.append("Verify approach compatibility with existing system architecture")
        
        if session["blocked_decisions"] > 0:
            recommendations.append("Ensure knowledge consultation before attempting decisions")
        
        return recommendations
    
    def _save_consultation_log(self):
        """Save consultation log to persistent storage."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "enforcement_mode": self.enforcement_mode,
            "active_sessions": len(self.agent_sessions),
            "sessions": {k: {**v, "mcp_tools_used": list(v["mcp_tools_used"])} 
                        for k, v in self.agent_sessions.items()}
        }
        
        os.makedirs(os.path.dirname(self.consultation_log), exist_ok=True)
        with open(self.consultation_log, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def get_system_metrics(self) -> Dict:
        """Get system-wide metrics on knowledge consultation compliance."""
        total_sessions = len(self.agent_sessions)
        if total_sessions == 0:
            return {"message": "No active sessions"}
        
        compliant_sessions = sum(1 for s in self.agent_sessions.values() if s["knowledge_consulted"])
        total_decisions = sum(s["decisions_made"] for s in self.agent_sessions.values())
        total_blocked = sum(s["blocked_decisions"] for s in self.agent_sessions.values())
        
        return {
            "system_compliance": {
                "total_sessions": total_sessions,
                "compliant_sessions": compliant_sessions,
                "compliance_rate": (compliant_sessions / total_sessions) * 100.0,
                "target_compliance_rate": 90.0
            },
            "decision_enforcement": {
                "total_decisions": total_decisions,
                "blocked_decisions": total_blocked,
                "approval_rate": ((total_decisions - total_blocked) / max(total_decisions, 1)) * 100.0
            },
            "knowledge_utilization": {
                "agents_using_knowledge": len([s for s in self.agent_sessions.values() 
                                             if len(s["mcp_tools_used"]) > 0]),
                "average_tools_per_session": sum(len(s["mcp_tools_used"]) for s in self.agent_sessions.values()) / max(total_sessions, 1)
            }
        }


# Global enforcer instance for system-wide use
_global_enforcer = None

def get_knowledge_enforcer(enforcement_mode="strict"):
    """Get the global knowledge consultation enforcer instance."""
    global _global_enforcer
    if _global_enforcer is None:
        _global_enforcer = KnowledgeConsultationEnforcer(enforcement_mode)
    return _global_enforcer


def main():
    """Demonstrate knowledge consultation enforcement system."""
    print("🔒 Knowledge Consultation Enforcement System")
    print("=" * 60)
    
    # Create enforcer
    enforcer = KnowledgeConsultationEnforcer("strict")
    
    # Simulate agent session
    session_key = enforcer.start_agent_session("rif-analyst", "113", "Analyze knowledge usage")
    
    # Try to make decision without knowledge consultation (should be blocked)
    print("\n🔍 Testing decision without knowledge consultation...")
    decision_allowed = enforcer.request_decision_approval(
        session_key, 
        "analysis_recommendation", 
        "Recommend solution approach"
    )
    
    print(f"Decision allowed: {decision_allowed}")
    
    # Simulate knowledge consultation
    print("\n📚 Simulating knowledge consultation...")
    enforcer.record_knowledge_consultation(
        session_key,
        "mcp__rif-knowledge__get_claude_documentation",
        "claude code capabilities",
        {"result": "Claude Code documentation retrieved"}
    )
    
    enforcer.record_knowledge_consultation(
        session_key,
        "mcp__rif-knowledge__query_knowledge", 
        "knowledge consultation patterns",
        {"patterns": ["consultation-enforcement", "decision-blocking"]}
    )
    
    # Try decision again (should be allowed now)
    print("\n🔍 Testing decision after knowledge consultation...")
    decision_allowed = enforcer.request_decision_approval(
        session_key,
        "analysis_recommendation", 
        "Recommend solution approach"
    )
    
    print(f"Decision allowed: {decision_allowed}")
    
    # Generate report
    print("\n📊 Session Report:")
    report = enforcer.generate_consultation_report(session_key)
    print(json.dumps(report, indent=2))
    
    # System metrics
    print("\n📈 System Metrics:")
    metrics = enforcer.get_system_metrics()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()