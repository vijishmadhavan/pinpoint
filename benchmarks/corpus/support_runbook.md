# Support Runbook

High queue depth procedure:
1. Check current queue depth and worker error rate.
2. Pause low-priority backfills if queue depth is above 50,000 jobs.
3. Restart stalled consumers only after confirming the database is healthy.
4. Notify the on-call engineer and incident commander in the operations channel.

Weekly operations checklist:
- Review failed jobs older than 24 hours.
- Verify backup completion.
- Rotate API keys only during the maintenance window.

Customer escalation process:
- Acknowledge severity-one incidents within 15 minutes.
- Provide status updates every 30 minutes until mitigation.
