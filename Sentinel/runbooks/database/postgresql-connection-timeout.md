---
title: Resolve PostgreSQL Connection Timeout
service: postgresql
error_type: connection_timeout
severity: high
tags:
  - database
  - postgresql
  - connection
  - timeout
author: Database Team
last_updated: 2024-02-20
version: "1.3"
---

# Resolve PostgreSQL Connection Timeout

## Overview

This runbook addresses PostgreSQL connection timeout issues where applications fail to establish connections to the database within the expected time frame.

## Symptoms

- Application logs show: `connection timed out` or `could not connect to server`
- Error: `FATAL: too many connections for role`
- Slow or hanging database queries
- Connection pool exhaustion warnings
- Increased latency in database-dependent services

## Common Root Causes

1. **Connection Pool Exhaustion**: Max connections reached
2. **Network Issues**: Firewall, DNS, or routing problems
3. **Database Overload**: High CPU/memory usage on DB server
4. **Long-Running Queries**: Blocking other connections
5. **Idle Connections**: Zombie connections holding slots
6. **Configuration Issues**: Incorrect connection string or timeouts

## Diagnosis Steps

### Step 1: Check Database Connectivity

```bash
# Test basic connectivity
psql -h <host> -U <user> -d <database> -c "SELECT 1;"

# With timeout
PGCONNECT_TIMEOUT=5 psql -h <host> -U <user> -d <database> -c "SELECT 1;"
```

### Step 2: Check Connection Count

```sql
-- Current connections
SELECT count(*) FROM pg_stat_activity;

-- Connections by state
SELECT state, count(*) 
FROM pg_stat_activity 
GROUP BY state;

-- Connections by application
SELECT application_name, count(*) 
FROM pg_stat_activity 
GROUP BY application_name;
```

### Step 3: Check Max Connections

```sql
SHOW max_connections;

-- Compare with current
SELECT count(*) as current, 
       (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max
FROM pg_stat_activity;
```

### Step 4: Identify Long-Running Queries

```sql
SELECT pid, now() - pg_stat_activity.query_start AS duration, query, state
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
ORDER BY duration DESC;
```

### Step 5: Check for Blocking Queries

```sql
SELECT blocked_locks.pid AS blocked_pid,
       blocking_locks.pid AS blocking_pid,
       blocked_activity.query AS blocked_query,
       blocking_activity.query AS blocking_query
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

### Step 6: Check Database Server Resources

```bash
# CPU and Memory (on DB server)
top -b -n 1 | head -20

# Disk I/O
iostat -x 1 5

# Check PostgreSQL logs
tail -100 /var/log/postgresql/postgresql-*-main.log
```

## Resolution Steps

### If Connection Pool Exhaustion

1. **Terminate idle connections**:
   ```sql
   SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE state = 'idle' 
   AND query_start < now() - interval '10 minutes';
   ```

2. **Increase max_connections** (requires restart):
   ```sql
   ALTER SYSTEM SET max_connections = 200;
   -- Then restart PostgreSQL
   ```

3. **Review application connection pool settings** - reduce pool size per app

### If Long-Running Queries

1. **Kill the blocking query**:
   ```sql
   SELECT pg_terminate_backend(<pid>);
   ```

2. **Cancel the query** (graceful):
   ```sql
   SELECT pg_cancel_backend(<pid>);
   ```

### If Network Issues

1. Check firewall rules allow port 5432
2. Verify DNS resolution: `nslookup <db-host>`
3. Check network route: `traceroute <db-host>`
4. Verify security groups (cloud environments)

### If Database Overload

1. Check for missing indexes on slow queries
2. Consider read replicas for read-heavy workloads
3. Scale up database resources (CPU/RAM)
4. Enable connection pooling (PgBouncer)

## Verification

1. Test connectivity from application servers
2. Monitor connection count over time
3. Check application logs for successful connections
4. Verify query latency returns to normal

## Prevention

- Use connection pooling (PgBouncer, pgpool-II)
- Set appropriate connection timeouts in applications
- Monitor connection count with alerts
- Regular query optimization and index maintenance
- Implement circuit breakers in applications
- Set statement_timeout to prevent runaway queries

## Related Runbooks

- [PostgreSQL High CPU](./postgresql-high-cpu.md)
- [PostgreSQL Replication Lag](./postgresql-replication-lag.md)
- [Database Failover Procedure](./postgresql-failover.md)
