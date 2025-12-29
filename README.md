# Auditory semantic 1-back pilot

This repository contains a PsychoPy (Builder) experiment for a pilot fMRI study
testing an auditory semantic 1-back task with concrete and abstract words.

The experiment is intended to validate task feasibility, timing, and data logging prior to application in target populations.

---

## Experiment overview

- Modality: auditory (spoken words)
- Design: block design
- Conditions:
  - Concrete words (CON)
  - Abstract words (ABS)
  - Auditory baseline (distorted speech)
- Task: semantic 1-back comparison
- Total duration: ~10–12 minutes

---

## Structure

- `experiment/` - PsychoPy Builder experiment (`.psyexp`) and resources  
- `experiment/resources/` - audio stimuli and CSV lists  
- `instructions/` - task instructions  
- `data/` - output data (excluded from version control)

---

## Requirements

- PsychoPy **2023.2.3** (standalone)
- Windows OS

See `ENVIRONMENT.md` for detailed environment specification.

---

## Running the experiment

Open the '.psyexp' file in PsychoPy Builder and run it using the PsychoPy
standalone application.

---

## Status

- Environment verified
- Experiment under active development

