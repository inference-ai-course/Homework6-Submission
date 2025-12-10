"""
Conversation State Manager
Manages conversation history and session state in memory
"""
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
import logging
logger = logging.getLogger(__name__)


class ConversationStateManager:
    """Manages conversation state for multiple sessions"""
    
    def __init__(self, max_history: int = 10, session_timeout_minutes: int = 30):
        """
        Initialize the state manager
        
        Args:
            max_history: Maximum number of conversation turns to keep
            session_timeout_minutes: Session timeout in minutes
        """
        self.max_history = max_history
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.sessions: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        
    def create_session(self, session_id: str) -> Dict:
        """
        Create a new conversation session
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session data dictionary
        """

        self.sessions[session_id] = {
            "conversation_history": [],
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        logger.info(f"Created session: {self.sessions[session_id]}")
        return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get session data
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                # Check if session has expired
                if datetime.now() - session["last_activity"] > self.session_timeout:
                    del self.sessions[session_id]
                    return None
                return session
            return None
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to the conversation history
        
        Args:
            session_id: Session identifier
            role: Message role (user or assistant)
            content: Message content
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                session = self.create_session(session_id)
            session["conversation_history"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            # Keep only the most recent messages
            if len(session["conversation_history"]) > self.max_history * 2:  # *2 for user+assistant pairs
                session["conversation_history"] = session["conversation_history"][-self.max_history * 2:]
            
            session["last_activity"] = datetime.now()
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation messages
        """
        session = self.get_session(session_id)
        if session:
            return session["conversation_history"]
        return []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a conversation session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cleared, False if not found
        """
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions
        
        Returns:
            Number of sessions removed
        """
        with self.lock:
            expired_sessions = []
            current_time = datetime.now()
            
            for session_id, session in self.sessions.items():
                if current_time - session["last_activity"] > self.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
            
            return len(expired_sessions)
    
    def get_active_sessions_count(self) -> int:
        """
        Get the number of active sessions
        
        Returns:
            Number of active sessions
        """
        with self.lock:
            return len(self.sessions)
