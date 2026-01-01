---
title: Restart Kubernetes Pod in CrashLoopBackOff
service: kubernetes
error_type: CrashLoopBackOff
severity: high
tags:
  - kubernetes
  - pods
  - restart
  - crashloop
author: SRE Team
last_updated: 2024-01-15
version: "2.1"
---

# Restart Kubernetes Pod in CrashLoopBackOff

## Overview

This runbook addresses Kubernetes pods stuck in `CrashLoopBackOff` status. This occurs when a container repeatedly crashes after starting, causing Kubernetes to back off and wait longer between restart attempts.

## Symptoms

- Pod shows `CrashLoopBackOff` status in `kubectl get pods`
- Container exit code is non-zero
- Events show repeated container restarts
- Increasing restart count in pod status
- Exponential backoff delay between restarts (10s, 20s, 40s, etc.)

## Common Root Causes

1. **Application Error**: The application crashes during startup
2. **Configuration Error**: Missing or invalid environment variables/ConfigMaps
3. **Resource Limits**: Container hits memory or CPU limits
4. **Health Check Failure**: Liveness probe fails repeatedly
5. **Missing Dependencies**: Required services not available
6. **Permission Issues**: Container can't access required files/secrets

## Diagnosis Steps

### Step 1: Check Pod Status

```bash
kubectl get pod <pod-name> -n <namespace>
kubectl describe pod <pod-name> -n <namespace>
```

Look for:
- Restart count
- Last state and exit code
- Events section for errors

### Step 2: Check Container Logs

```bash
# Current container logs
kubectl logs <pod-name> -n <namespace>

# Previous container logs (before crash)
kubectl logs <pod-name> -n <namespace> --previous
```

### Step 3: Check Events

```bash
kubectl get events -n <namespace> --sort-by='.lastTimestamp' | grep <pod-name>
```

### Step 4: Verify Resource Limits

```bash
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].resources}'
```

Check if the container is hitting memory/CPU limits.

### Step 5: Check ConfigMaps and Secrets

```bash
kubectl get configmap -n <namespace>
kubectl get secrets -n <namespace>
```

Verify all required ConfigMaps and Secrets exist.

## Resolution Steps

### If Application Error

1. Review application logs from `--previous` flag
2. Check if recent deployments introduced bugs
3. Consider rolling back: `kubectl rollout undo deployment/<name>`

### If Resource Limits

1. Increase memory/CPU limits in deployment spec
2. Apply changes: `kubectl apply -f deployment.yaml`
3. Or patch directly:
   ```bash
   kubectl patch deployment <name> -p '{"spec":{"template":{"spec":{"containers":[{"name":"<container>","resources":{"limits":{"memory":"512Mi"}}}]}}}}'
   ```

### If Configuration Error

1. Verify environment variables in deployment
2. Check ConfigMap values: `kubectl get configmap <name> -o yaml`
3. Update and apply corrected configuration

### If Health Check Failure

1. Review liveness probe configuration
2. Increase `initialDelaySeconds` if app needs more startup time
3. Increase `timeoutSeconds` or `failureThreshold`

### Force Restart (After Fixing Root Cause)

```bash
# Delete pod (deployment will recreate it)
kubectl delete pod <pod-name> -n <namespace>

# Or restart entire deployment
kubectl rollout restart deployment/<deployment-name> -n <namespace>
```

## Verification

1. Confirm pod is running: `kubectl get pod <pod-name> -n <namespace>`
2. Check logs for normal operation: `kubectl logs <pod-name> -n <namespace> -f`
3. Verify application health endpoints
4. Monitor for 5-10 minutes to ensure stability

## Prevention

- Implement proper health checks
- Set appropriate resource limits based on actual usage
- Use init containers for dependency checks
- Implement graceful shutdown handling
- Add proper error handling and logging in applications

## Related Runbooks

- [Pod OOMKilled](./kubernetes-oomkilled.md)
- [Pod ImagePullBackOff](./kubernetes-imagepullbackoff.md)
- [Deployment Rollback](./kubernetes-rollback.md)
