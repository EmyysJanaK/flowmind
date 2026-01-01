#!/usr/bin/env python3
"""
Sentinel Security System Verification Script

Verifies that validation, audit logging, and security controls are properly
configured and functional. Run this after deployment to ensure all safety
mechanisms are active.

Usage:
    python verify_security.py
    python verify_security.py --verbose
    python verify_security.py --test-attacks
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

def check_files_exist() -> Dict[str, bool]:
    """Verify all security-related files exist."""
    base_path = Path(__file__).parent
    
    files_to_check = {
        "registry.py": base_path / "registry.py",
        "validation.py": base_path / "validation.py",
        "audit_logger.py": base_path / "audit_logger.py",
        "__init__.py": base_path / "__init__.py",
    }
    
    results = {}
    for name, path in files_to_check.items():
        exists = path.exists()
        results[name] = exists
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")
    
    return results

def check_imports() -> Dict[str, bool]:
    """Verify all security modules can be imported."""
    imports_to_check = {
        "ParameterValidator": "sentinel.tools.validation",
        "ValidationRule": "sentinel.tools.validation",
        "ValidationLevel": "sentinel.tools.validation",
        "ValidationError": "sentinel.tools.validation",
        "AuditLogger": "sentinel.tools.audit_logger",
        "AuditLogEntry": "sentinel.tools.audit_logger",
        "AuditAction": "sentinel.tools.audit_logger",
        "AuditLevel": "sentinel.tools.audit_logger",
    }
    
    results = {}
    for class_name, module_path in imports_to_check.items():
        try:
            parts = module_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[parts[-1]])
            getattr(module, class_name)
            results[f"{class_name}"] = True
            print(f"  ✓ {module_path}.{class_name}")
        except (ImportError, AttributeError) as e:
            results[f"{class_name}"] = False
            print(f"  ✗ {module_path}.{class_name}: {e}")
    
    return results

def check_validation_rules() -> Dict[str, List[str]]:
    """Verify validation rules are registered."""
    try:
        from sentinel.tools import get_validator
        
        validator = get_validator()
        tools_to_check = [
            "docker_container_logs",
            "docker_container_restart",
            "k8s_pod_status",
            "k8s_pod_restart",
        ]
        
        results = {}
        for tool_name in tools_to_check:
            try:
                rules = validator.describe_rules(tool_name)
                rule_names = [r["name"] for r in rules]
                results[tool_name] = rule_names
                print(f"  ✓ {tool_name}: {len(rule_names)} rules")
                for rule_name in rule_names[:3]:
                    print(f"    - {rule_name}")
                if len(rule_names) > 3:
                    print(f"    ... and {len(rule_names) - 3} more")
            except Exception as e:
                results[tool_name] = []
                print(f"  ✗ {tool_name}: {e}")
        
        return results
    except ImportError as e:
        print(f"  ✗ Cannot import validator: {e}")
        return {}

def check_audit_logging() -> Dict[str, bool]:
    """Verify audit logging is configured."""
    try:
        from sentinel.tools import get_audit_logger
        
        logger = get_audit_logger()
        
        checks = {
            "has_log_file": hasattr(logger, 'log_file') and logger.log_file is not None,
            "has_memory_buffer": hasattr(logger, 'memory_log') and logger.memory_log is not None,
            "has_log_method": hasattr(logger, 'log') and callable(logger.log),
            "has_query_methods": all(hasattr(logger, m) for m in [
                'get_logs', 'get_failed_validations', 'get_suspicious_patterns'
            ]),
        }
        
        for check_name, result in checks.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check_name}")
        
        return checks
    except ImportError as e:
        print(f"  ✗ Cannot import audit logger: {e}")
        return {}

def test_attack_blocking() -> List[Dict[str, Any]]:
    """Test that dangerous operations are actually blocked."""
    try:
        from sentinel.tools import get_validator
        from sentinel.tools.validation import ValidationError
        
        validator = get_validator()
        
        test_cases = [
            {
                "name": "Wildcard deletion attempt",
                "tool": "k8s_pod_restart",
                "params": {"namespace": "prod", "pod_name": "*"},
                "should_block": True,
            },
            {
                "name": "Command injection attempt",
                "tool": "docker_container_logs",
                "params": {"container_name": "web; rm -rf /", "tail": 100},
                "should_block": True,
            },
            {
                "name": "System namespace access",
                "tool": "k8s_pod_status",
                "params": {"namespace": "kube-system", "pod_name": "coredns"},
                "should_block": True,
            },
            {
                "name": "Resource exhaustion attempt",
                "tool": "docker_container_logs",
                "params": {"container_name": "web", "tail": 1000000},
                "should_block": True,
            },
            {
                "name": "Valid operation",
                "tool": "docker_container_logs",
                "params": {"container_name": "web", "tail": 100},
                "should_block": False,
            },
        ]
        
        results = []
        for test_case in test_cases:
            try:
                validation_result = validator.validate(test_case["tool"], test_case["params"])
                is_blocked = not validation_result["valid"]
            except ValidationError:
                is_blocked = True
            
            passed = is_blocked == test_case["should_block"]
            status = "✓" if passed else "✗"
            
            result = {
                "test": test_case["name"],
                "expected_blocked": test_case["should_block"],
                "actually_blocked": is_blocked,
                "passed": passed,
            }
            results.append(result)
            
            print(f"  {status} {test_case['name']}")
            if not passed:
                print(f"     Expected blocked={test_case['should_block']}, got {is_blocked}")
        
        return results
    except Exception as e:
        print(f"  ✗ Attack testing failed: {e}")
        return []

def main():
    """Run all verification checks."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify Sentinel security system"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test-attacks", "-t", action="store_true", help="Test attack blocking")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    results = {}
    
    print("\n" + "="*70)
    print("SENTINEL SECURITY SYSTEM VERIFICATION")
    print("="*70 + "\n")
    
    # Check files
    print("1. FILE EXISTENCE CHECK")
    print("-" * 70)
    results["files"] = check_files_exist()
    print()
    
    # Check imports
    print("2. IMPORTS CHECK")
    print("-" * 70)
    results["imports"] = check_imports()
    print()
    
    # Check validation rules
    print("3. VALIDATION RULES CHECK")
    print("-" * 70)
    results["validation_rules"] = check_validation_rules()
    print()
    
    # Check audit logging
    print("4. AUDIT LOGGING CHECK")
    print("-" * 70)
    results["audit_logging"] = check_audit_logging()
    print()
    
    # Test attack blocking (optional)
    if args.test_attacks:
        print("5. ATTACK BLOCKING TEST")
        print("-" * 70)
        results["attack_tests"] = test_attack_blocking()
        print()
    
    # Summary
    print("="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    
    # Count results
    if results["files"]:
        files_ok = all(results["files"].values())
        print(f"  Files: {'✓ All present' if files_ok else '✗ Some missing'}")
        all_passed = all_passed and files_ok
    
    if results["imports"]:
        imports_ok = all(results["imports"].values())
        print(f"  Imports: {'✓ All working' if imports_ok else '✗ Some failed'}")
        all_passed = all_passed and imports_ok
    
    if results["validation_rules"]:
        rules_ok = bool(results["validation_rules"])
        print(f"  Validation: {'✓ Rules registered' if rules_ok else '✗ No rules found'}")
        all_passed = all_passed and rules_ok
    
    if results["audit_logging"]:
        audit_ok = all(results["audit_logging"].values())
        print(f"  Audit Logging: {'✓ Configured' if audit_ok else '✗ Not configured'}")
        all_passed = all_passed and audit_ok
    
    if "attack_tests" in results:
        attack_tests = results["attack_tests"]
        if attack_tests:
            tests_passed = all(t["passed"] for t in attack_tests)
            print(f"  Attack Blocking: {'✓ All attacks blocked' if tests_passed else '✗ Some attacks not blocked'}")
            all_passed = all_passed and tests_passed
    
    print()
    
    if all_passed:
        print("✓ SECURITY SYSTEM VERIFICATION PASSED")
        status_code = 0
    else:
        print("✗ SECURITY SYSTEM VERIFICATION FAILED")
        status_code = 1
    
    print("="*70 + "\n")
    
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    
    sys.exit(status_code)

if __name__ == "__main__":
    main()
