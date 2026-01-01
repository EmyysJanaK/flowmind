---
title: Resolve Container OOMKilled
service: docker
error_type: OOMKilled
severity: critical
tags:
  - docker
  - memory
  - oom
  - container
author: Platform Team
last_updated: 2024-03-10
version: "1.0"
---

# Resolve Container OOMKilled

## Overview

This runbook addresses containers being killed by the Linux Out-of-Memory (OOM) killer due to exceeding memory limits. OOMKilled is a critical issue that causes service disruption.

## Symptoms

- Container exit code 137 (128 + 9 = SIGKILL)
- Docker/Kubernetes shows `OOMKilled` reason
- `dmesg` shows OOM killer messages
- Application suddenly terminates without graceful shutdown
- Memory usage graphs show spike before crash

## Common Root Causes

1. **Memory Leak**: Application gradually consumes more memory
2. **Insufficient Limits**: Memory limit too low for workload
3. **Traffic Spike**: Sudden increase in requests/data
4. **Large Data Processing**: Batch jobs loading too much into memory
5. **JVM Heap Configuration**: Java heap larger than container limit
6. **Caching Issues**: Unbounded in-memory caches

## Diagnosis Steps

### Step 1: Confirm OOMKilled

```bash
# Docker
docker inspect <container-id> | grep -A 5 "State"

# Kubernetes
kubectl describe pod <pod-name> | grep -A 5 "Last State"
```

Look for `OOMKilled: true` or exit code 137.

### Step 2: Check System OOM Events

```bash
# On the host
dmesg | grep -i "oom\|killed process"

# Recent kernel messages
journalctl -k | grep -i oom
```

### Step 3: Analyze Memory Usage Pattern

```bash
# Docker stats (live)
docker stats <container-id>

# Kubernetes metrics
kubectl top pod <pod-name>
```

### Step 4: Check Container Memory Limits

```bash
# Docker
docker inspect <container-id> | grep -A 10 "Memory"

# Kubernetes
kubectl get pod <pod-name> -o jsonpath='{.spec.containers[*].resources}'
```

### Step 5: Check Application Memory

For Java applications:
```bash
# Get heap dump before crash (if possible)
jmap -dump:format=b,file=heap.hprof <pid>

# Check heap usage
jstat -gc <pid>
```

For Node.js:
```bash
# Check heap size
node --v8-options | grep -i heap
```

## Resolution Steps

### Immediate Actions

1. **Restart the container** with current limits:
   ```bash
   docker restart <container-id>
   # or
   kubectl delete pod <pod-name>
   ```

2. **Increase memory limits** (temporary fix):
   ```bash
   # Docker
   docker update --memory="1g" --memory-swap="1g" <container-id>
   
   # Kubernetes - update deployment
   kubectl patch deployment <name> -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"memory":"1Gi"}}}]}}}}'
   ```

### Long-Term Fixes

#### For Memory Leaks

1. Enable heap profiling in development
2. Use memory profilers:
   - Java: VisualVM, YourKit, async-profiler
   - Node.js: clinic.js, heapdump
   - Python: memory_profiler, tracemalloc
3. Review recent code changes for leaks
4. Check for circular references, unclosed resources

#### For JVM Applications

Ensure JVM respects container limits:
```bash
# Modern JVM (11+)
java -XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0 -jar app.jar

# Older JVM
java -Xmx768m -Xms768m -jar app.jar
```

Set heap to ~75% of container memory limit to leave room for:
- Native memory
- Thread stacks
- Direct buffers
- Metaspace

#### For Unbounded Caches

1. Set maximum cache size
2. Implement LRU eviction
3. Use external caching (Redis)

#### Right-Size Memory Limits

1. Monitor actual usage over time
2. Set limits at p99 usage + 20% buffer
3. Set requests at average usage

## Verification

1. Container stays running after restart
2. Memory usage stabilizes below limit
3. No OOM events in `dmesg`
4. Application metrics show healthy memory pattern

## Prevention

- Always set memory limits in production
- Configure JVM/runtime to respect container limits
- Implement graceful memory pressure handling
- Set up memory usage alerts at 80% of limit
- Regular load testing with memory monitoring
- Use circuit breakers to shed load under pressure

## Related Runbooks

- [Container High Memory Usage](./docker-high-memory.md)
- [Java Heap Analysis](./java-heap-analysis.md)
- [Kubernetes Resource Quotas](./kubernetes-resource-quotas.md)
