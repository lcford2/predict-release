---
fontsize: 10pt
geometry:
- margin=0.5in
---

## Notes from Meeting with Ximing

Attendees: Sankar Arumugam, Ximing Cai, Donghui Li, Shiqi Fang, Lucas Ford

Last Modified: {{currentdate}}

## Donghui Presentation

- Objective: Improve reservoir operation of NWM using their model
- Using a Data-based reservoir operation model (DROM)
  - Inputs
    - Daily inflow
    - Storage
    - PDSI
    - Day of year
  - Output
    - Daily release
  - Decision tree structure
    - Convert to if else for implementation
  - 250 Reservoirs across CONUS each reservoir has a set of rules
- Going to do
  - Write operation module on DROM to supplement level-pool scheme in NWM
  - Use learned tree if res is in current 250, else use original level-pool scheme
- Planned NWM addition
  - write new module that implements decision trees for 250 reservoirs
  - modify MWN IO to allow NWM to read and process the required PDSI input
