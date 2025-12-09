"""
Test script for Voice Assistant components
Run this to verify individual services are working
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from services.whisper_service import WhisperService
from services.llama_service import LLaMAService
from services.speecht5_service import SpeechT5Service
from services.state_manager import ConversationStateManager


def test_state_manager():
    """Test conversation state manager"""
    print("\n" + "="*50)
    print("Testing Conversation State Manager")
    print("="*50)
    
    manager = ConversationStateManager(max_history=5, session_timeout_minutes=30)
    
    # Create session
    session_id = "test-session-123"
    manager.create_session(session_id)
    print(f"✓ Created session: {session_id}")
    
    # Add messages
    manager.add_message(session_id, "user", "Hello!")
    manager.add_message(session_id, "assistant", "Hi! How can I help you?")
    print("✓ Added messages to conversation")
    
    # Get history
    history = manager.get_conversation_history(session_id)
    print(f"✓ Retrieved history: {len(history)} messages")
    
    for msg in history:
        print(f"  - {msg['role']}: {msg['content']}")
    
    # Clear session
    manager.clear_session(session_id)
    print("✓ Cleared session")
    
    print("\n✅ State Manager tests passed!")


def test_whisper_service():
    """Test Whisper transcription service"""
    print("\n" + "="*50)
    print("Testing Whisper Service")
    print("="*50)
    
    try:
        service = WhisperService(model_name="base")
        print("✓ Whisper service initialized")
        
        service.load_model()
        print("✓ Whisper model loaded successfully")
        
        print("\n✅ Whisper service tests passed!")
        print("Note: Actual transcription test requires an audio file")
        
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")


def test_llama_service():
    """Test LLaMA conversation service"""
    print("\n" + "="*50)
    print("Testing LLaMA Service")
    print("="*50)
    
    try:
        # Note: This will take time and require model download
        print("⚠️  LLaMA test requires model download and GPU/CPU resources")
        print("⚠️  Skipping actual model loading for quick test")
        
        service = LLaMAService()
        print("✓ LLaMA service initialized")
        
        print("\n✅ LLaMA service tests passed (initialization only)")
        print("Note: Full test requires model download and resources")
        
    except Exception as e:
        print(f"❌ LLaMA test failed: {e}")


def test_speecht5_service():
    """Test SpeechT5 TTS service"""
    print("\n" + "="*50)
    print("Testing SpeechT5 Service")
    print("="*50)
    
    try:
        service = SpeechT5Service()
        print("✓ SpeechT5 service initialized")
        
        # Test synthesis
        print("⚠️  SpeechT5 test requires model download")
        print("⚠️  Skipping actual synthesis for quick test")
        
        print("\n✅ SpeechT5 service tests passed (initialization only)")
        print("Note: Full test requires model download")
        
    except Exception as e:
        print(f"❌ SpeechT5 test failed: {e}")


def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("🧪 Voice Assistant Component Tests")
    print("="*50)
    
    test_state_manager()
    test_whisper_service()
    test_llama_service()
    test_speecht5_service()
    
    print("\n" + "="*50)
    print("✅ All tests completed!")
    print("="*50)


if __name__ == "__main__":
    main()
