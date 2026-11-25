# Camera Full Frame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Restore the complete top-camera image, preserve existing 180-degree orientation, explain shared B-button recording, and document confirmed gripper endpoints.

**Architecture:** User clarified that existing rotation is correct. Preview and collector retain flip=True for top; the collector converts full RGB to BGR without crop/resize. Dataset metadata describes image geometry and gripper endpoints.

**Tech Stack:** Python, OpenCV, configparser, pytest; existing Windows environment.

**Spec:** User's current request and two onsite photographs in this chat; 0=closed, 255=open. Direct implementation is already authorized in this chat.

## Global Constraints

- Keep current teleoperation mapping, motion protection and gripper command scale.
- Preserve existing recorded debug files; cropped pixels cannot be recovered.
- No robot commands for verification.

## Review Focus

- Four distinct frame corners survive collector JPEG output (test collector).
- Default 180-degree rotation remains intact (test driver and startup).
- Preview and collector pass identical top orientation (test startup call sites).
- Switching from right B to left B stops the shared recording (test actual button monitor).
- Both B releases in one polling sample toggle once (test actual button monitor).

## Task 1: Full-frame camera and explicit orientation

Files: `experiments/run_control.py`; existing camera driver and preview receive regression coverage.

Interface: existing `flip=True` for top remains; driver keeps `read()` returning RGB.

- [x] Add failing tests for complete collector corners and preserved preview/startup orientation.
- [x] Run `pytest tests/test_camera_recording_controls.py -q` and verify expected failures.
- [x] Remove top crop and resize, use full RGB-to-BGR conversion; retain original flip=True.
- [x] Rerun camera tests and ACT/teleop/gripper regressions.

## Task 2: B-button behavior and endpoint documentation

Files: new camera/control tests, collector metadata, ACT manual and change record.

Interface: existing `button_monitor_realtime(agent, stop_event, ready_event)`; no new button semantics.

- [x] Test sequential cross-hand releases, simultaneous releases, and servo-off ignored presses.
- [x] Describe one shared recording session and 0=closed/255=open in metadata/manual; keep numeric commands unchanged.
- [x] Run focused suite, inspect diff and deliver restart/preview steps. No commit/push requested.

Completion: user corrected initial orientation assumption before delivery; no camera settings/INI rotation changes retained. Mixed image geometries are rejected by converter.
