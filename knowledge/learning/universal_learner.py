#!/usr/bin/env python3
"""
Universal Learner - Automatic Pattern Extraction from Claude Code Sessions

This learner runs on EVERY tool use via hooks and automatically:
- Tracks patterns across Edit, Bash, Read actions
- Identifies error/fix patterns
- Detects file relationships
- Saves patterns for auto-ingestion

No manual intervention required - learns from actual work patterns.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import hashlib
from typing import Dict, Any, List, Optional
import re

class UniversalLearner:
    """Learns patterns from every Claude Code action automatically."""
    
    def __init__(self):
        self.knowledge_path = Path("/Users/cal/DEV/RIF/knowledge")
        self.session_file = Path("/tmp/claude_session.json")
        self.events_file = self.knowledge_path / "events.jsonl"
        self.patterns_dir = self.knowledge_path / "patterns" / "auto_learned"
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize session
        self.session = self.load_session()
        
    def load_session(self) -> Dict[str, Any]:
        """Load current session context."""
        if self.session_file.exists():
            try:
                return json.loads(self.session_file.read_text())
            except:
                pass
        return {
            "start_time": datetime.now().isoformat(),
            "recent_files": [],
            "recent_commands": [],
            "recent_errors": [],
            "edit_counts": {},
            "error_fixes": [],
            "file_sequences": []
        }
    
    def save_session(self):
        """Save session context."""
        self.session_file.write_text(json.dumps(self.session, indent=2))
    
    def learn_from_action(self):
        """Main entry point - called on every tool use."""
        # Get environment variables from Claude hook
        tool_name = os.environ.get('CLAUDE_TOOL_NAME', '')
        tool_arg1 = os.environ.get('CLAUDE_TOOL_ARG1', '')
        exit_code = os.environ.get('CLAUDE_TOOL_EXIT_CODE', '0')
        stderr = os.environ.get('CLAUDE_TOOL_STDERR', '')
        
        # Skip if no tool info
        if not tool_name:
            return
        
        # Log event
        self.log_event({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "arg": tool_arg1,
            "exit_code": exit_code,
            "error": stderr if exit_code != '0' else None
        })
        
        # Learn based on tool type
        if tool_name in ['Edit', 'Write', 'MultiEdit']:
            self.learn_edit_pattern(tool_arg1)
        elif tool_name == 'Bash':
            if exit_code == '0':
                self.learn_success_pattern(tool_arg1)
            else:
                self.learn_error_pattern(tool_arg1, stderr)
        elif tool_name == 'Read':
            self.track_exploration(tool_arg1)
        
        # Detect cross-tool patterns
        self.detect_sequences()
        
        # Save session
        self.save_session()
    
    def learn_edit_pattern(self, file_path: str):
        """Learn from file edits."""
        if not file_path:
            return
        
        # Track edit frequency
        self.session["edit_counts"][file_path] = self.session["edit_counts"].get(file_path, 0) + 1
        
        # Add to recent files
        if file_path not in self.session["recent_files"]:
            self.session["recent_files"].append(file_path)
        
        # Keep only last 20 files
        self.session["recent_files"] = self.session["recent_files"][-20:]
        
        # Detect hot spots (files edited 5+ times)
        if self.session["edit_counts"][file_path] >= 5:
            self.create_pattern({
                "pattern_name": f"hotspot-{Path(file_path).name}-{datetime.now().strftime('%Y%m%d')}",
                "pattern_type": "hotspot",
                "description": f"File {file_path} is frequently modified ({self.session['edit_counts'][file_path]} times)",
                "file": file_path,
                "edit_count": self.session["edit_counts"][file_path],
                "related_files": self.find_related_files(file_path),
                "recommendation": "Consider refactoring or adding tests for stability",
                "auto_learned": True,
                "confidence": min(0.5 + (self.session["edit_counts"][file_path] * 0.1), 1.0)
            })
        
        # Detect file relationships
        if len(self.session["recent_files"]) >= 3:
            recent = self.session["recent_files"][-3:]
            if len(set(recent)) == 3:  # 3 different files
                self.session["file_sequences"].append(recent)
                
                # If this sequence appears 3+ times, it's a pattern
                sequence_key = tuple(sorted(recent))
                sequence_count = sum(1 for seq in self.session["file_sequences"] 
                                    if tuple(sorted(seq)) == sequence_key)
                
                if sequence_count >= 3:
                    self.create_pattern({
                        "pattern_name": f"file-relationship-{hashlib.md5(str(sequence_key).encode()).hexdigest()[:8]}",
                        "pattern_type": "file_relationship",
                        "description": "These files are commonly edited together",
                        "files": list(sequence_key),
                        "occurrence_count": sequence_count,
                        "recommendation": "Consider these files as a logical unit",
                        "auto_learned": True,
                        "confidence": min(0.6 + (sequence_count * 0.1), 1.0)
                    })
    
    def learn_error_pattern(self, command: str, error: str):
        """Learn from command errors."""
        if not command:
            return
        
        # Track error
        error_entry = {
            "command": command,
            "error": error[:500],  # Limit error message length
            "timestamp": datetime.now().isoformat(),
            "context_files": self.session["recent_files"][-3:] if self.session["recent_files"] else []
        }
        
        self.session["recent_errors"].append(error_entry)
        self.session["recent_errors"] = self.session["recent_errors"][-10:]  # Keep last 10
        
        # Check if this error was fixed
        if len(self.session["recent_commands"]) >= 2:
            last_success = None
            for cmd in reversed(self.session["recent_commands"]):
                if cmd.get("exit_code") == "0" and self.similar_command(cmd["command"], command):
                    last_success = cmd["command"]
                    break
            
            if last_success:
                # We found a fix!
                self.create_pattern({
                    "pattern_name": f"error-fix-{hashlib.md5((command + last_success).encode()).hexdigest()[:8]}",
                    "pattern_type": "error_fix",
                    "description": "Command error and successful fix",
                    "problem": {
                        "command": command,
                        "error": self.extract_error_key(error)
                    },
                    "solution": {
                        "command": last_success,
                        "change": self.diff_commands(command, last_success)
                    },
                    "context_files": error_entry["context_files"],
                    "auto_learned": True,
                    "confidence": 0.8
                })
    
    def learn_success_pattern(self, command: str):
        """Learn from successful commands."""
        if not command:
            return
        
        # Track successful command
        self.session["recent_commands"].append({
            "command": command,
            "exit_code": "0",
            "timestamp": datetime.now().isoformat()
        })
        self.session["recent_commands"] = self.session["recent_commands"][-20:]
        
        # Detect command sequences
        if len(self.session["recent_commands"]) >= 3:
            # Check for repeated successful sequences
            recent_cmds = [c["command"] for c in self.session["recent_commands"][-3:]]
            cmd_pattern = self.extract_command_pattern(recent_cmds)
            
            if cmd_pattern:
                self.create_pattern({
                    "pattern_name": f"command-sequence-{hashlib.md5(str(cmd_pattern).encode()).hexdigest()[:8]}",
                    "pattern_type": "command_sequence",
                    "description": "Successful command sequence pattern",
                    "commands": cmd_pattern,
                    "context": "Commands that work well together",
                    "auto_learned": True,
                    "confidence": 0.7
                })
    
    def track_exploration(self, file_path: str):
        """Track file exploration patterns."""
        # This helps understand how problems are investigated
        if file_path and len(self.session["recent_files"]) > 0:
            last_file = self.session["recent_files"][-1] if self.session["recent_files"] else None
            if last_file and Path(last_file).parent == Path(file_path).parent:
                # Exploring files in same directory - likely investigating related code
                pass  # Could create investigation patterns here
    
    def detect_sequences(self):
        """Detect patterns across multiple actions."""
        # Look for Read -> Edit patterns (understanding before modifying)
        # Look for Bash fail -> Edit -> Bash success patterns (debugging)
        # This is where cross-tool intelligence emerges
        pass
    
    def create_pattern(self, pattern: Dict[str, Any]):
        """Create a pattern file for auto-ingestion."""
        # Add metadata
        pattern["created_at"] = datetime.now().isoformat()
        pattern["created_by"] = "universal_learner"
        pattern["source"] = "automatic_learning"
        
        # Generate unique filename
        pattern_id = hashlib.md5(json.dumps(pattern, sort_keys=True).encode()).hexdigest()[:12]
        pattern_file = self.patterns_dir / f"pattern_{pattern_id}.json"
        
        # Don't create duplicate patterns
        if not pattern_file.exists():
            pattern_file.write_text(json.dumps(pattern, indent=2))
            
            # Log pattern creation
            self.log_event({
                "event": "pattern_created",
                "pattern_name": pattern["pattern_name"],
                "pattern_type": pattern["pattern_type"],
                "confidence": pattern.get("confidence", 0.5)
            })
    
    def log_event(self, event: Dict[str, Any]):
        """Log event to events.jsonl."""
        with open(self.events_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def find_related_files(self, file_path: str) -> List[str]:
        """Find files commonly edited with this file."""
        related = []
        for f in self.session["recent_files"]:
            if f != file_path and self.session["edit_counts"].get(f, 0) > 2:
                related.append(f)
        return related[:5]  # Top 5 related files
    
    def similar_command(self, cmd1: str, cmd2: str) -> bool:
        """Check if two commands are similar."""
        # Remove common variations
        cmd1_clean = re.sub(r'\s+', ' ', cmd1).strip()
        cmd2_clean = re.sub(r'\s+', ' ', cmd2).strip()
        
        # Same base command?
        if cmd1_clean.split()[0] == cmd2_clean.split()[0]:
            # Calculate similarity
            return self.string_similarity(cmd1_clean, cmd2_clean) > 0.6
        return False
    
    def string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity (0-1)."""
        longer = max(len(s1), len(s2))
        if longer == 0:
            return 1.0
        
        # Simple character-based similarity
        matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
        return matches / longer
    
    def extract_error_key(self, error: str) -> str:
        """Extract key error message."""
        # Look for common error patterns
        patterns = [
            r"Error: (.+?)(?:\n|$)",
            r"Exception: (.+?)(?:\n|$)",
            r"AttributeError: (.+?)(?:\n|$)",
            r"TypeError: (.+?)(?:\n|$)",
            r"SyntaxError: (.+?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error)
            if match:
                return match.group(1)[:100]
        
        # Return first line if no pattern matches
        return error.split('\n')[0][:100]
    
    def diff_commands(self, cmd1: str, cmd2: str) -> str:
        """Find the difference between two commands."""
        words1 = cmd1.split()
        words2 = cmd2.split()
        
        if words1[0] != words2[0]:
            return f"Changed command from {words1[0]} to {words2[0]}"
        
        # Find first difference
        for i, (w1, w2) in enumerate(zip(words1, words2)):
            if w1 != w2:
                return f"Changed '{w1}' to '{w2}'"
        
        if len(words1) != len(words2):
            return f"Added/removed arguments"
        
        return "Minor changes"
    
    def extract_command_pattern(self, commands: List[str]) -> Optional[List[str]]:
        """Extract pattern from command sequence."""
        if len(commands) < 2:
            return None
        
        # Look for common command types
        base_commands = [cmd.split()[0] for cmd in commands]
        
        # Common development sequences
        known_sequences = [
            ["git", "status"],
            ["python", "python"],  # Running scripts multiple times
            ["npm", "test"],
            ["cargo", "build"],
            ["pytest", "pytest"],
        ]
        
        for seq in known_sequences:
            if all(bc in seq for bc in base_commands[:len(seq)]):
                return commands
        
        # If commands are related (same base), consider it a pattern
        if len(set(base_commands)) <= 2:
            return commands
        
        return None


def main():
    """Main entry point for hook execution."""
    learner = UniversalLearner()
    learner.learn_from_action()


if __name__ == "__main__":
    main()