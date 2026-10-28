-- Builds data/Azure_Data0.csv from the Azure trace database (SQLite).
-- One machine, priority 0 jobs, and jobs that finished within the trace.
.headers on
.mode csv
.output Azure_Data0.csv

SELECT vm.vmId, vm.vmTypeId, vmType.machineID, vm.starttime, vm.endtime,
       vmType.core, vmType.memory, vmType.ssd, vmType.nic, vm.Priority
FROM vm
JOIN vmType ON vm.vmTypeId = vmType.vmTypeId
WHERE vmType.machineID = 0
  AND vm.Priority = 0
  AND vm.starttime > 0
  AND vm.endtime IS NOT NULL;
.output stdout