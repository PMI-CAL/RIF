#!/usr/bin/env python3
"""
Session Tracker - Maintains context across Claude Code actions

This lightweight tracker maintains session state that the universal learner
can use to detect patterns. It's called frequently so it's optimized for speed.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class SessionTracker:
    """Fast session context tracker."""
    
    def __init__(self):
        self.session_file = Path("/tmp/claude_session.json")
        self.max_age_hours = 24  # Clean up old sessions
        
    def get_or_create_session(self) -> Dict[str, Any]:
        """Get current session or create new one."""
        session = self.load_session()
        
        # Check if session is too old
        if session:
            start_time = datetime.fromisoformat(session.get("start_time", datetime.now().isoformat()))
            if datetime.now() - start_time > timedelta(hours=self.max_age_hours):
                # Create fresh session
                session = None
        
        if not session:
            session = {
                "start_time": datetime.now().isoformat(),
                "session_id": os.getpid(),  # Use PID as simple session ID
                "recent_files": [],
                "recent_commands": [],
                "recent_errors": [],
                "edit_counts": {},
                "action_count": 0
            }
            self.save_session(session)
        
        return session
    
    def load_session(self) -> Optional[Dict[str, Any]]:
        """Load session from file."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    return json.load(f)
            except:
                # Corrupted session file
                self.session_file.unlink(missing_ok=True)
        return None
    
    def save_session(self, session: Dict[str, Any]):
        """Save session to file."""
        # Increment action count
        session["action_count"] = session.get("action_count", 0) + 1
        
        # Limit list sizes to prevent unbounded growth
        session["recent_files"] = session["recent_files"][-20:]
        session["recent_commands"] = session["recent_commands"][-20:]
        session["recent_errors"] = session["recent_errors"][-10:]
        
        # Write atomically
        temp_file = self.session_file.with_suffix('.tmp')
        temp_file.write_text(json.dumps(session, indent=2))
        temp_file.replace(self.session_file)
    
    def add_file(self, file_path: str, session: Dict[str, Any]):
        """Add file to recent files."""
        if file_path and file_path not in session["recent_files"]:
            session["recent_files"].append(file_path)
    
    def add_command(self, command: str, exit_code: str, session: Dict[str, Any]):
        """Add command to recent commands."""
        session["recent_commands"].append({
            "command": command,
            "exit_code": exit_code,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_error(self, command: str, error: str, session: Dict[str, Any]):
        """Add error to recent errors."""
        session["recent_errors"].append({
            "command": command,
            "error": error[:500],  # Limit size
            "timestamp": datetime.now().isoformat()
        })
    
    def increment_edit_count(self, file_path: str, session: Dict[str, Any]):
        """Increment edit count for a file."""
        session["edit_counts"][file_path] = session.get("edit_counts", {}).get(file_path, 0) + 1
    
    def cleanup_old_sessions(self):
        """Clean up old session files."""
        # Clean up other temp files if needed
        temp_dir = Path("/tmp")
        for old_session in temp_dir.glob("claude_session_*.json"):
            try:
                # Check age
                age = datetime.now() - datetime.fromtimestamp(old_session.stat().st_mtime)
                if age > timedelta(hours=48):
                    old_session.unlink()
            except:
                pass


def quick_track():
    """Quick tracking function for hooks."""
    # This is called from hooks, so it needs to be fast
    tracker = SessionTracker()
    session = tracker.get_or_create_session()
    
    # Get environment variables
    tool_name = os.environ.get('CLAUDE_TOOL_NAME', '')
    tool_arg1 = os.environ.get('CLAUDE_TOOL_ARG1', '')
    exit_code = os.environ.get('CLAUDE_TOOL_EXIT_CODE', '0')
    stderr = os.environ.get('CLAUDE_TOOL_STDERR', '')
    
    # Quick updates based on tool
    if tool_name in ['Edit', 'Write', 'MultiEdit']:
        tracker.add_file(tool_arg1, session)
        tracker.increment_edit_count(tool_arg1, session)
    elif tool_name == 'Bash':
        tracker.add_command(tool_arg1, exit_code, session)
        if exit_code != '0':
            tracker.add_error(tool_arg1, stderr, session)
    elif tool_name == 'Read':
        tracker.add_file(tool_arg1, session)
    
    # Save session
    tracker.save_session(session)
    
    # Occasional cleanup
    if session["action_count"] % 100 == 0:
        tracker.cleanup_old_sessions()


if __name__ == "__main__":
    quick_track()