"""
scratch/run_all_tests.py
========================
Helper script to run all 9 test suites sequentially and output a final pass/fail matrix.
"""

import subprocess
import sys
import os

TEST_COMMANDS = [
    ("test_state.py", "python test_state.py"),
    ("test_router.py", "python tests/test_router.py"),
    ("test_memory.py", "python tests/test_memory.py"),
    ("test_classifier.py", "python tests/test_classifier.py"),
    ("test_rag.py", "python tests/test_rag.py"),
    ("test_supervisor.py", "python tests/test_supervisor.py"),
    ("test_hitl.py", "python tests/test_hitl.py"),
    ("test_memory_recall_integration.py", "python tests/test_memory_recall_integration.py"),
    ("test_graph.py", "python tests/test_graph.py")
]

def run_cmd(cmd):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    res = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout, res.stderr

def main():
    print("=" * 60)
    print("RUNNING ALL TEST SUITES SEQUENTIALLY")
    print("=" * 60)
    
    results = []
    for name, cmd in TEST_COMMANDS:
        print(f"Running {name}...")
        code, out, err = run_cmd(cmd)
        status = "PASS" if code == 0 else "FAIL"
        results.append((name, status, out, err))
        print(f"Finished {name}: {status}")
        
    print("\n" + "=" * 60)
    print("TEST EXECUTION MATRIX")
    print("=" * 60)
    for name, status, _, _ in results:
        print(f"{name:<40} : {status}")
        
    failures = [name for name, status, _, _ in results if status == "FAIL"]
    if failures:
        print(f"\nWARNING: Some tests failed: {failures}")
        # Print failure details
        for name, status, out, err in results:
            if status == "FAIL":
                print(f"\n--- {name} stdout ---")
                print(out)
                print(f"--- {name} stderr ---")
                print(err)
        sys.exit(1)
    else:
        print("\nAll 9 test suites passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
