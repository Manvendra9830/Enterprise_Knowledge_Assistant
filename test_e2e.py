"""
End-to-End Test Suite for Enterprise Knowledge Assistant
Tests Retrieval, Guardrails, Conversation Memory, Fallback Modes, and Upload/Delete.
"""

import httpx
import os
import sys
import time

API_URL = "http://localhost:8000/api"

def print_result(name, passed, details=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} | {name} {details}")

def test_health():
    print("\n--- Testing Health Endpoint ---")
    try:
        r = httpx.get(f"{API_URL}/health", timeout=120.0)
        data = r.json()
        print(f"Status: {r.status_code}")
        print(f"Gemini: {data.get('gemini')} | Ollama: {data.get('ollama')}")
        print_result("Health Check", r.status_code == 200)
        return True
    except Exception as e:
        print_result("Health Check", False, f"Exception: {e}")
        return False

def test_retrieval():
    print("\n--- Testing Retrieval ---")
    queries = [
        "What is the annual leave policy?",
        "What is the refund policy?",
        "What is the notice period?"
    ]
    for q in queries:
        try:
            r = httpx.post(f"{API_URL}/ask", json={"question": q, "session_id": "test_retrieval"}, timeout=120.0)
            data = r.json()
            sources = data.get("sources", [])
            passed = r.status_code == 200 and len(sources) > 0
            print_result(f"Retrieval: '{q}'", passed, f"({len(sources)} sources cited)")
        except Exception as e:
            print_result(f"Retrieval: '{q}'", False, f"Exception: {e}")

def test_guardrails():
    print("\n--- Testing Guardrails ---")
    queries = [
        "Who won FIFA World Cup 2022?",
        "What is the capital of France?"
    ]
    for q in queries:
        try:
            r = httpx.post(f"{API_URL}/ask", json={"question": q, "session_id": "test_guardrails"}, timeout=120.0)
            data = r.json()
            ans = data.get("answer", "")
            sources = data.get("sources", [])
            passed = r.status_code == 200 and len(sources) == 0 and "I don't have enough information" in ans
            print_result(f"Guardrails: '{q}'", passed)
        except Exception as e:
            print_result(f"Guardrails: '{q}'", False, f"Exception: {e}")

def test_memory():
    print("\n--- Testing Conversation Memory ---")
    
    # Upload Pihu Context since Pihu_CV.pdf was deleted earlier
    pihu_file = "pihu_context.txt"
    with open(pihu_file, "w") as f:
        f.write("Pihu is a software engineer. Pihu studies at Madras Institute of Technology.")
    try:
        with open(pihu_file, "rb") as f:
            httpx.post(f"{API_URL}/upload", files={"file": ("pihu_context.txt", f)}, timeout=120.0)
    except Exception:
        pass
        
    session_id = "test_memory_session_1"
    
    # Q1
    q1 = "Who is Pihu?"
    try:
        r1 = httpx.post(f"{API_URL}/ask", json={"question": q1, "session_id": session_id}, timeout=120.0)
        passed1 = r1.status_code == 200 and len(r1.json().get("sources", [])) > 0
        print_result("Memory Q1 (Context setting)", passed1)
    except Exception as e:
        print_result("Memory Q1 (Context setting)", False, str(e))
        
    # Q2
    q2 = "Where does she study?"
    try:
        r2 = httpx.post(f"{API_URL}/ask", json={"question": q2, "session_id": session_id}, timeout=120.0)
        ans = r2.json().get("answer", "").lower()
        passed2 = r2.status_code == 200 and ("madras" in ans or "institute" in ans)
        print_result("Memory Q2 (Pronoun Resolution)", passed2)
    except Exception as e:
        print_result("Memory Q2 (Pronoun Resolution)", False, str(e))
        
    # Cleanup Pihu Context
    try:
        httpx.delete(f"{API_URL}/documents/pihu_context.txt", timeout=120.0)
        os.remove(pihu_file)
    except Exception:
        pass

def test_upload_delete():
    print("\n--- Testing Upload and Delete ---")
    # Upload
    test_file_path = "test_dummy.txt"
    with open(test_file_path, "w") as f:
        f.write("This is a test dummy document for E2E testing.")
        
    try:
        with open(test_file_path, "rb") as f:
            r = httpx.post(f"{API_URL}/upload", files={"file": ("test_dummy.txt", f)}, timeout=120.0)
        passed_upload = r.status_code == 200 and r.json().get("status") == "success"
        print_result("Document Upload", passed_upload)
    except Exception as e:
        print_result("Document Upload", False, str(e))
        
    # Delete
    try:
        r = httpx.delete(f"{API_URL}/documents/test_dummy.txt", timeout=120.0)
        passed_delete = r.status_code == 200 and r.json().get("status") == "deleted"
        print_result("Document Delete", passed_delete)
    except Exception as e:
        print_result("Document Delete", False, str(e))
        
    if os.path.exists(test_file_path):
        os.remove(test_file_path)

if __name__ == "__main__":
    print("Starting E2E Tests...\n")
    test_health()
    test_retrieval()
    test_guardrails()
    test_memory()
    test_upload_delete()
    print("\nE2E Tests Completed.")
