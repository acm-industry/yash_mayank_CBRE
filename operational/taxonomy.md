# CBRE Facilities Dispatch — Call Intake Taxonomy (SOP)

**Document owner:** Facilities Operations
**Audience:** intake operators, dispatch coordinators, QA reviewers
**Revision:** 2024-11 (internal)

This is the call-intake taxonomy used by the night-desk and day-desk intake
teams. When a tenant calls in, an operator triages the issue into one of
the categories described below, picks an appropriate risk tier, jots a short
note, and opens a ticket in the CMMS.

Reclassification after dispatch is allowed — the on-site technician's
finding is authoritative. The original intake record stays on the ticket
for review, and the technician's category becomes the resolved category.

---

## Call categories

CBRE Facilities groups every incoming call into one of ten top-level
categories. Each category covers a handful of finer-grained issue types;
the exact subtype labels are shared across all CMMS records (you'll see
them in the historical exports).

**PLUMBING** — anything water- or pipe-related: leaks (supply, drain, or
roof / skylight intrusion), failing restroom fixtures, and drainage or
sewer backups.

**ELECTRICAL** — power, lighting, and electrical hazards. Covers outages,
lighting issues (flickering, dim, burned-out), unsafe conditions like
sparking outlets or hot panels, and low-voltage / data problems (network,
phones, structured cabling).

**HVAC** — climate and air. Cooling and heating failures, air-quality
complaints (odors, fumes, ventilation, suspected mold), refrigerant leaks
on chillers and AC, and issues isolated to thermostats or building
controls.

**ELEVATOR** — anything elevator-related, from people stuck inside a
stopped car down to minor cosmetic complaints (mis-leveling, slow doors,
noises).

**DOORS_ACCESS** — physical access points: mechanical doors, broken or
damaged glass, badge / keypad / electronic-lock problems, and powered
automatic doors.

**LIFE_SAFETY** — situations involving danger to people: confirmed gas
or chemical incidents, fire or smoke with visual confirmation, trip /
slip / spill hazards, and structural failures like ceiling collapse or
falling debris.

**SECURITY** — people-related concerns: suspicious individuals,
breaches or trespassing, and active threats.

**JANITORIAL** — routine cleanliness. Restroom supplies, carpet and
floor spills, and trash or odor complaints (non-chemical).

**GROUNDS_EXTERIOR** — outdoor and perimeter issues: parking-lot
lighting, sidewalk and pavement damage, signage and fencing.

**PEST_SPECIALTY** — the catch-all bucket: pest infestations, break-room
appliances, and landscaping or irrigation.

The exact subtype labels and the conventions for which ones land where
are visible across the historical corpus. When in doubt about how to
sort a particular call, look at how prior similar calls were classified.

---

## Risk levels

Intake operators stamp every call with one of four risk tiers. Dispatch
uses the tier to set urgency.

- **LOW** — routine maintenance, no safety or operational impact.
  Typical: a burned-out lightbulb, an empty soap dispenser.
- **MEDIUM** — localized issue affecting a single tenant or area;
  needs attention within the shift.
  Typical: one tenant's AC is out, a clogged restroom.
- **HIGH** — meaningful safety, operational, or financial risk if it
  waits more than a few hours.
  Typical: an active leak that's spreading, power out on one floor.
- **EMERGENCY** — immediate threat to life, property, or business
  continuity. Dispatch right away; duty-manager is notified in parallel.
  Typical: confirmed gas leak, visible fire, person stuck in an
  elevator, an active threat.

These are guideline tiers — what counts as "meaningful risk" or
"few hours" is judgment. The historical corpus is the best reference
for how prior calls were tiered.

---

## Asking a follow-up question

If a safety-critical detail is missing, the operator may ask one targeted
follow-up — but only one, and only when it materially changes how the
call is handled. Most calls don't need a follow-up, and asking unnecessary
questions slows throughput and frustrates callers.

---

## Operator note shorthand

Intake notes are free text typed fast under time pressure. Across the
historical exports you'll see common abbreviations: floors as `F9` or
`9F`, suites as `Ste 404`, `wtr` for water, `elec` for electrical,
`intermit` for intermittent, `w/` and `w/o` for with / without, and
operator initials at the end of the line (`—dw`, `—tc`). Conventions
aren't strict — different shifts have their own habits.

---

## Reclassification & QA review

When an on-site technician finds a different root cause than what was
logged at intake, the resolved category is updated on the ticket. The
original intake record stays intact for review.

CBRE's QA team samples past tickets weekly and writes notes on what
intake got wrong — over-classifications, mis-classifications, wrong
floors. Those notes live in a separate review file alongside the
historical corpus.
